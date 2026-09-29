//! PyO3 bindings, compiled as the Python extension module `semantic_world._core`.
//!
//! The module exposes one class, `World`, which loads an experiment configuration, resets
//! to a seed, and steps. Observations cross as NumPy arrays; manifests, events, and info
//! cross as plain dictionaries and lists.

mod session;

use std::collections::BTreeMap;
use std::path::Path;

use numpy::{IntoPyArray, ndarray};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde_json::Value;
use sw_rules::ActionChoice;
use sw_schema::{Resolved, load_experiment_with_overrides};
use sw_sense::{BlockData, Observation};

use session::{Session, SessionError};

/// Version of the Rust crates, from the workspace `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Names of the engine crates linked into this module, in build-stage order.
pub const CRATES: [&str; 6] = [
    sw_schema::CRATE_NAME,
    sw_core::CRATE_NAME,
    sw_rules::CRATE_NAME,
    sw_sense::CRATE_NAME,
    sw_render::CRATE_NAME,
    sw_log::CRATE_NAME,
];

/// Mistakes in what the caller passed are `ValueError`s; anything else is a `RuntimeError`.
fn py_err(e: SessionError) -> PyErr {
    use sw_rules::RuleError;
    match e {
        SessionError::Usage(m) => PyValueError::new_err(m),
        SessionError::Schema(e) => PyValueError::new_err(e.to_string()),
        SessionError::Rules(RuleError::Usage(m)) => PyValueError::new_err(m),
        SessionError::Rules(e @ RuleError::NotAllowed { .. }) => {
            PyValueError::new_err(e.to_string())
        }
        SessionError::Core(e @ sw_core::CoreError::UnknownAgent(_)) => {
            PyValueError::new_err(e.to_string())
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// A JSON value as the matching Python object.
fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any()
            }
        }
        Value::String(s) => s.into_pyobject(py)?.into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.into_any()
        }
    })
}

/// A Python object as a JSON value: None, bools, ints, floats, strings, lists, and dicts
/// with string keys.
fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::from(u));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::from(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            map.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::new();
        for item in list.iter() {
            items.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(items));
    }
    Err(PyValueError::new_err(format!(
        "cannot convert {} to a configuration value",
        obj.get_type().name()?
    )))
}

fn observation_to_py<'py>(py: Python<'py>, obs: &Observation) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let blocks = PyDict::new(py);
    for block in &obs.blocks {
        let shape = ndarray::IxDyn(&block.shape);
        let array: Bound<'py, PyAny> = match &block.data {
            BlockData::F32(v) => ndarray::ArrayD::from_shape_vec(shape, v.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any(),
            BlockData::U8(v) => ndarray::ArrayD::from_shape_vec(shape, v.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any(),
        };
        blocks.set_item(&block.name, array)?;
    }
    dict.set_item("blocks", blocks)?;
    dict.set_item("vector", obs.vector.clone().into_pyarray(py))?;
    match &obs.props {
        Some(facts) => {
            let list = PyList::empty(py);
            for fact in facts {
                list.append(fact.to_string())?;
            }
            dict.set_item("props", list)?;
        }
        None => dict.set_item("props", py.None())?,
    }
    match &obs.mask {
        Some(mask) => dict.set_item("mask", mask.clone().into_pyarray(py))?,
        None => dict.set_item("mask", py.None())?,
    }
    dict.set_item("text", py.None())?;
    Ok(dict)
}

fn observations_to_py<'py>(
    py: Python<'py>,
    obs: &BTreeMap<String, Observation>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (agent, observation) in obs {
        dict.set_item(agent, observation_to_py(py, observation)?)?;
    }
    Ok(dict)
}

/// An action written as a dictionary: `{"type": "move", "direction": 0.1, "speed": 0.5}` or
/// `{"type": "eat", "target": "berry_bush_3"}`.
fn choice_from_py(action: &Bound<'_, PyAny>) -> PyResult<ActionChoice> {
    let dict = action.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err("an action is a dict with a \"type\" key and the action's arguments")
    })?;
    let name: String = dict
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("an action needs a \"type\""))?
        .extract()?;
    let mut choice = ActionChoice::named(&name);
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        if key == "type" {
            continue;
        }
        if key == "target" {
            if !value.is_none() {
                choice.target = Some(value.extract::<String>()?);
            }
            continue;
        }
        let number: f64 = value.extract().map_err(|_| {
            PyValueError::new_err(format!("action argument `{key}` must be a number"))
        })?;
        choice.numbers.insert(key, number);
    }
    Ok(choice)
}

fn events_to_py<'py>(py: Python<'py>, events: &[sw_rules::Event]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for event in events {
        let value =
            serde_json::to_value(event).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        list.append(json_to_py(py, &value)?)?;
    }
    Ok(list)
}

/// A world: an experiment configuration, reset to a seed and stepped.
#[pyclass(name = "World", module = "semantic_world._core")]
struct PyWorld {
    resolved: Resolved,
    session: Option<Session>,
    seed: Option<u64>,
}

impl PyWorld {
    fn session(&self) -> PyResult<&Session> {
        self.session
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call reset() before using the world"))
    }

    fn session_mut(&mut self) -> PyResult<&mut Session> {
        self.session
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("call reset() before using the world"))
    }
}

#[pymethods]
impl PyWorld {
    /// Load an experiment configuration, with optional overrides keyed by dotted path.
    #[new]
    #[pyo3(signature = (config, overrides=None))]
    fn new(config: &str, overrides: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut pairs = Vec::new();
        if let Some(overrides) = overrides {
            for (k, v) in overrides.iter() {
                pairs.push((k.extract::<String>()?, py_to_json(&v)?));
            }
        }
        let resolved = load_experiment_with_overrides(Path::new(config), &pairs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyWorld {
            resolved,
            session: None,
            seed: None,
        })
    }

    /// Start a run. Without a seed, the configuration's `seed` or first of `seeds` is used.
    /// Returns `(observations, info)`.
    #[pyo3(signature = (seed=None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyAny>)> {
        let seed = seed
            .or_else(|| self.resolved.experiment.master_seeds().first().copied())
            .ok_or_else(|| PyValueError::new_err("the configuration has no seed"))?;
        let mut session = Session::new(&self.resolved, seed).map_err(py_err)?;
        let observations = session.observe().map_err(py_err)?;
        let info = json_to_py(py, &session.info())?;
        self.seed = Some(seed);
        self.session = Some(session);
        Ok((observations_to_py(py, &observations)?, info))
    }

    /// One decision: `actions` maps agent IDs to action dicts. Returns
    /// `(observations, events, info)`. No reward: reward is computed on the agent's side.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyDict>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyList>, Bound<'py, PyAny>)> {
        let mut choices = BTreeMap::new();
        for (agent, action) in actions.iter() {
            choices.insert(agent.extract::<String>()?, choice_from_py(&action)?);
        }
        let session = self.session_mut()?;
        let (observations, events) = session.step(&choices).map_err(py_err)?;
        let info = json_to_py(py, &session.info())?;
        Ok((
            observations_to_py(py, &observations)?,
            events_to_py(py, &events)?,
            info,
        ))
    }

    /// Every agent's ID, in creation order.
    #[getter]
    fn agents(&self) -> PyResult<Vec<String>> {
        Ok(self.session()?.agents())
    }

    /// The agents that must act before the next step.
    #[getter]
    fn agents_awaiting_action(&self) -> PyResult<Vec<String>> {
        Ok(self.session()?.agents_awaiting_action())
    }

    #[getter]
    fn done(&self) -> PyResult<bool> {
        Ok(self.session()?.done())
    }

    #[getter]
    fn tick(&self) -> PyResult<u64> {
        Ok(self.session()?.world().clock().tick)
    }

    #[getter]
    fn time_s(&self) -> PyResult<f64> {
        Ok(self.session()?.world().clock().time_s())
    }

    /// The master seed of the current run.
    #[getter]
    fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// The sensor and action manifests of every agent:
    /// `{agent: {"sensors": ..., "actions": ...}}`.
    fn manifests<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let session = self.session()?;
        let dict = PyDict::new(py);
        for agent in session.agents() {
            let (sensors, actions) = session.manifests(&agent).map_err(py_err)?;
            let entry = PyDict::new(py);
            let sensors = serde_json::to_value(&sensors)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let actions = serde_json::to_value(&actions)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            entry.set_item("sensors", json_to_py(py, &sensors)?)?;
            entry.set_item("actions", json_to_py(py, &actions)?)?;
            dict.set_item(agent, entry)?;
        }
        Ok(dict)
    }

    /// The 32-byte seed of the agent's `agent:<id>` stream.
    fn agent_seed<'py>(&self, py: Python<'py>, agent: &str) -> PyResult<Bound<'py, PyBytes>> {
        let seed = self
            .session()?
            .world()
            .agent_seed(agent)
            .map_err(|e| PyKeyError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &seed))
    }

    /// The full canonical state. **Privileged:** for scripted agents, tests, and logs only.
    /// Learning agents must never read it.
    fn debug_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let snapshot = self.session()?.world().snapshot();
        let value =
            serde_json::to_value(&snapshot).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }

    /// The state hash at this tick, as hex.
    fn state_hash(&self) -> PyResult<String> {
        Ok(self.session()?.world().state_hash().to_hex())
    }

    /// The fully resolved experiment configuration, as loaded.
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.resolved.experiment_value)
    }
}

/// The version of the Rust crates behind this module.
#[pyfunction]
fn version() -> &'static str {
    VERSION
}

/// The names of the engine crates linked into this module.
#[pyfunction]
fn crates() -> Vec<&'static str> {
    CRATES.to_vec()
}

/// Whether the module was built with the `rerun` Cargo feature.
#[pyfunction]
fn rerun_enabled() -> bool {
    sw_log::rerun_enabled()
}

/// The `semantic_world._core` module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(crates, m)?)?;
    m.add_function(wrap_pyfunction!(rerun_enabled, m)?)?;
    m.add_class::<PyWorld>()?;
    Ok(())
}
