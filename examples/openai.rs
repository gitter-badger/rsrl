extern crate rsrl;

use rsrl::{run, Parameter, SerialExperiment, Evaluation};
use rsrl::fa::linear::RBFNetwork;
use rsrl::agents::control::td::QSigma;
use rsrl::domains::{Domain, open_ai};
use rsrl::policies::{Greedy, EpsilonGreedy};
use rsrl::geometry::Space;

use rsrl::logging;


fn main() {
    let domain = open_ai::Env::new("CartPole-v1").unwrap();

    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let q_func = RBFNetwork::new(domain.state_space().partitioned(8), n_actions);
        let policy = EpsilonGreedy::new(aspace, Parameter::exponential(0.5, 0.01, 0.95));

        QSigma::new(q_func, policy, 0.1, 0.99, 0.1, 2)
    };

    // Training:
    let logger = logging::stdout();
    let training_result = {
        let e = SerialExperiment::new(&mut agent,
                                      Box::new(|| open_ai::Env::new("CartPole-v1").unwrap()),
                                      1000);

        run(e, 1000, Some(logger))
    };

    // Testing:
    let testing_result =
        Evaluation::new(&mut agent,
                        Box::new(|| open_ai::Env::new("CartPole-v1").unwrap())).next().unwrap();


    println!("Solution \u{21D2} {} steps | reward {}",
             testing_result.steps,
             testing_result.reward);
}
