extern crate indicatif;
use self::indicatif::{ProgressBar, ProgressDrawTarget};

use agents::ControlAgent;
use domains::{Domain, Observation};
use geometry::{Space, ActionSpace};
use policies::Greedy;


/// Container for episodic statistics.
#[derive(Debug)]
pub struct Episode {
    /// The number of steps taken to reach the terminal state.
    pub n_steps: u64,

    /// The total accumulated reward over the episode.
    pub total_reward: f64
}


/// Helper function for running experiments.
pub fn run<T>(runner: T, n_episodes: usize) -> Vec<Episode>
    where T: Iterator<Item=Episode>
{
    let pb = ProgressBar::new(n_episodes as u64);
    pb.set_draw_target(ProgressDrawTarget::stdout());

    let out = runner.enumerate()
          .take(n_episodes)
          .inspect(|&(i, _)| {
              pb.set_position(i as u64);
          })
          .map(|(_, res)| res)
          .collect::<Vec<_>>();

    pb.finish_with_message("Training complete...");

    out
}


/// Utility for running a single evaluation episode.
pub struct Evaluation<'a, A: 'a, D> {
    agent: &'a mut A,
    domain_factory: Box<Fn() -> D>,

    greedy: Greedy,
}

impl<'a, S: Space, A, D> Evaluation<'a, A, D>
    where A: ControlAgent<S, ActionSpace>,
          D: Domain<StateSpace=S, ActionSpace=ActionSpace>
{
    pub fn new(agent: &'a mut A,
               domain_factory: Box<Fn() -> D>) -> Evaluation<'a, A, D>
    {
        Evaluation {
            agent: agent,
            domain_factory: domain_factory,

            greedy: Greedy,
        }
    }
}

impl<'a, S: Space, A, D> Iterator for Evaluation<'a, A, D>
    where A: ControlAgent<S, ActionSpace>,
          D: Domain<StateSpace=S, ActionSpace=ActionSpace>
{
    type Item = Episode;

    fn next(&mut self) -> Option<Episode> {
        let mut domain = (self.domain_factory)();
        let mut a = self.agent.evaluate_policy(&mut self.greedy,
                                               &domain.emit().state());

        let mut e = Episode {
            n_steps: 1,
            total_reward: 0.0,
        };

        loop {
            let t = domain.step(a);

            e.n_steps += 1;
            e.total_reward += t.reward;

            a = match t.to {
                Observation::Terminal(ref s) => {
                    self.agent.handle_terminal(s);
                    break;
                },
                _ => self.agent.evaluate_policy(&mut self.greedy,
                                                &t.to.state())
            };
        }

        Some(e)
    }
}


/// Utility for running a sequence of training episodes.
pub struct SerialExperiment<'a, A: 'a, D> {
    agent: &'a mut A,
    domain_factory: Box<Fn() -> D>,

    step_limit: u64
}

impl<'a, S: Space, A, D> SerialExperiment<'a, A, D>
    where A: ControlAgent<S, ActionSpace>,
          D: Domain<StateSpace=S, ActionSpace=ActionSpace>
{
    pub fn new(agent: &'a mut A,
               domain_factory: Box<Fn() -> D>,
               step_limit: u64) -> SerialExperiment<'a, A, D>
    {
        SerialExperiment {
            agent: agent,
            domain_factory: domain_factory,
            step_limit: step_limit,
        }
    }
}

impl<'a, S: Space, A, D> Iterator for SerialExperiment<'a, A, D>
    where A: ControlAgent<S, ActionSpace>,
          D: Domain<StateSpace=S, ActionSpace=ActionSpace>
{
    type Item = Episode;

    fn next(&mut self) -> Option<Episode> {
        let mut domain = (self.domain_factory)();
        let mut a = self.agent.pi(domain.emit().state());

        let mut e = Episode {
            n_steps: 1,
            total_reward: 0.0,
        };

        for j in 1..(self.step_limit+1) {
            let t = domain.step(a);

            e.n_steps = j;
            e.total_reward += t.reward;

            self.agent.handle_transition(&t);

            a = match t.to {
                Observation::Terminal(ref s) => {
                    self.agent.handle_terminal(s);
                    break;
                },
                _ => self.agent.pi(&t.to.state())
            };
        }

        Some(e)
    }
}
