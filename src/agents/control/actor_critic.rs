use Parameter;
use fa::QFunction;
use agents::{ControlAgent, PredictionAgent};
use domains::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, DifferentiablePolicy, Greedy};
use std::marker::PhantomData;


/// Regular gradient descent actor critic.
pub struct ActorCritic<S: Space, Q, C, P>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: Policy
{
    actor: Q,
    critic: C,

    policy: P,

    beta: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, Q, C, P> ActorCritic<S, Q, C, P>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: Policy
{
    pub fn new<T1, T2>(actor: Q, critic: C, policy: P,
                       beta: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        ActorCritic {
            actor: actor,
            critic: critic,

            policy: policy,

            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, Q, C, P> ControlAgent<S, ActionSpace> for ActorCritic<S, Q, C, P>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.actor.evaluate(s).as_slice())
    }

    fn evaluate_policy(&self, p: &mut Policy, s: &S::Repr) -> usize {
        p.sample(self.actor.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let td_error = self.critic.handle_transition(s, ns, t.reward).unwrap();

        self.actor.update_action(s, t.action, self.beta*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();
    }
}


/// Off-policy actor-critic algorithm.
///
/// http://icml.cc/2012/papers/268.pdf
pub struct OffPAC<S: Space, P, Q, C>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: DifferentiablePolicy
{
    actor: Q,
    critic: C,

    policy: P,

    alpha: Parameter,
    beta: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, P, Q, C> OffPAC<S, P, Q, C>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: DifferentiablePolicy
{
    pub fn new<T1, T2, T3>(actor: Q, critic: C, policy: P,
                           alpha: T1, beta: T2, gamma: T3) -> Self
    {
        OffPAC {
            actor: actor,
            critic: critic,

            policy: policy,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P, Q, C> ControlAgent<S, ActionSpace> for OffPAC<S, P, Q, C>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: DifferentiablePolicy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.actor.evaluate(s).as_slice())
    }

    fn evaluate_policy(&self, p: &mut Policy, s: &S::Repr) -> usize {
        p.sample(self.actor.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.actor.evaluate(s);

        let delta = t.reward +
            self.gamma*self.critic.evaluate(ns) - self.critic.evaluate(s);

        let rho = Greedy.probabilities(&qs)[t.action] /
            self.policy.probabilities(&qs)[t.action];

        self.actor.update_action(s, t.action, self.beta*delta);
        self.critic.update(s, self.alpha*delta);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();
    }
}
