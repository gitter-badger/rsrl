extern crate cpython;

use self::cpython::{Python, GILGuard, ObjectProtocol, NoArgs};
use self::cpython::{PyModule, PyObject, PyString, PyResult};

use super::{Observation, Transition, Domain};

use geometry::{ActionSpace, RegularSpace};
use geometry::dimensions::{Continuous, Discrete};


pub struct Gym {
    pub gil: GILGuard,
    pub gym: PyModule,
}

impl Gym {
    pub fn new() -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let gym = gil.python().import("gym")?;

        gil.python().run("import logging; logging.getLogger('gym.envs.registration').setLevel(logging.CRITICAL)", None, None)?;

        Ok(Self {
            gil: gil,
            gym: gym,
        })
    }

    fn py(&self) -> Python {
        self.gil.python()
    }

    fn make(&mut self, env_id: &str) -> PyResult<PyObject> {
        let py = self.gil.python();
        let maker = self.gym.get(py, "make")?;

        maker.call(py, (PyString::new(py, env_id),), None)
    }
}


pub struct Env {
    gym: Gym,
    env: PyObject,
    pub state: Vec<f64>,
    terminal: bool,
    last_reward: f64,
}

impl Env {
    pub fn new(env_id: &str) -> PyResult<Self> {
        let mut gym = Gym::new()?;
        let env = gym.make(env_id)?;

        let obs = env.call_method(gym.py(), "reset", NoArgs, None)?;
        let state = Env::parse_vec(gym.py(), &obs);

        Ok(Self {
            gym: gym,
            env: env,
            state: state,
            terminal: false,
            last_reward: 0.0,
        })
    }

    fn parse_vec(py: Python, vals: &PyObject) -> Vec<f64> {
        (0..vals.len(py).unwrap()).map(|i| {
            vals.get_item(py, i).unwrap().extract::<f64>(py).unwrap()
        }).collect()
    }

    pub fn update_state(&mut self, a: usize) {
        let py = self.gym.py();

        let tr = self.env.call_method(py, "step", (a,), None).unwrap();
        let obs = tr.get_item(py, 0).unwrap();

        self.state = Env::parse_vec(py, &obs);
        self.terminal = tr.get_item(py, 2).unwrap().extract::<bool>(py).unwrap();
        self.last_reward = tr.get_item(py, 1).unwrap().extract::<f64>(py).unwrap();
    }
}

impl Domain for Env {
    type StateSpace = RegularSpace<Continuous>;
    type ActionSpace = ActionSpace;

    fn emit(&self) -> Observation<Self::StateSpace, Self::ActionSpace> {
        if self.is_terminal() {
            Observation::Terminal(self.state.clone())
        } else {
            Observation::Full {
                state: self.state.clone(),
                actions: vec![]
            }
        }
    }

    fn step(&mut self, a: usize) -> Transition<Self::StateSpace, Self::ActionSpace> {
        let from = self.emit();

        self.update_state(a);
        let to = self.emit();

        Transition {
            from: from,
            action: a,
            reward: self.last_reward,
            to: to,
        }
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn reward(&self,
              _: &Observation<Self::StateSpace, Self::ActionSpace>,
              _: &Observation<Self::StateSpace, Self::ActionSpace>) -> f64
    {
        self.last_reward
    }

    fn state_space(&self) -> Self::StateSpace {
        let py = self.gym.py();
        let ss = self.env.getattr(py, "observation_space").unwrap();

        let lbs = ss.getattr(py, "low").unwrap();
        let ubs = ss.getattr(py, "high").unwrap();
        let len = ss.getattr(py, "shape").unwrap()
            .get_item(py, 0).unwrap()
            .extract::<usize>(py).unwrap();

        (0..len).fold(Self::StateSpace::new(), |acc, i| {
            let lb = lbs.get_item(py, i).unwrap().extract::<f64>(py).unwrap();
            let ub = ubs.get_item(py, i).unwrap().extract::<f64>(py).unwrap();

            acc.push(Continuous::new(lb, ub))
        })
    }

    fn action_space(&self) -> ActionSpace {
        let py = self.gym.py();
        let n = self.env
            .getattr(py, "action_space").unwrap()
            .getattr(py, "n").unwrap()
            .extract::<usize>(py).unwrap();

        ActionSpace::new(Discrete::new(n))
    }
}
