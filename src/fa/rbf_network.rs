use super::{Function, Parameterised, Linear, VFunction, QFunction};

use utils::{cartesian_product, dot};
use ndarray::{Axis, ArrayView, Array1, Array2};
use geometry::{Span, Space, RegularSpace};
use geometry::dimensions::{Partitioned, Continuous};


/// Optimised radial basis function network for function representation.
pub struct RBFNetwork {
    mu: Array2<f64>,
    gamma: Array1<f64>,

    weights: Array2<f64>,
}

impl RBFNetwork
{
    pub fn new(input_space: RegularSpace<Partitioned>, n_outputs: usize) -> Self
    {
        let n_features = match input_space.span() {
            Span::Finite(s) => s,
            _ =>
                panic!("`RBFNetwork` function approximator only supports \
                        partitioned input spaces.")
        };

        let centres = input_space.centres();
        let flat_combs =
            cartesian_product(&centres)
            .iter().cloned()
            .flat_map(|e| e).collect();

        RBFNetwork {
            mu: Array2::from_shape_vec((n_features, input_space.dim()), flat_combs).unwrap(),
            gamma: input_space.iter().map(|d| {
                let s = d.partition_width();
                -1.0 / (s * s)
            }).collect(),

            weights: Array2::<f64>::zeros((n_features, n_outputs)),
        }
    }
}


impl Function<Vec<f64>, f64> for RBFNetwork
{
    fn evaluate(&self, input: &Vec<f64>) -> f64 {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        <Self as VFunction<RegularSpace<Continuous>>>::evaluate_phi(self, &phi)
    }
}

impl Function<Vec<f64>, Vec<f64>> for RBFNetwork
{
    fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        // Apply matrix multiplication and return Vec<f64>:
        <Self as QFunction<RegularSpace<Continuous>>>::evaluate_phi(self, &phi)
    }
}


impl Parameterised<Vec<f64>, f64> for RBFNetwork
{
    fn update(&mut self, input: &Vec<f64>, error: f64) {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        <Self as VFunction<RegularSpace<Continuous>>>::update_phi(self, &phi, error);
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.mu == other.mu &&
            self.gamma == self.gamma &&
            self.weights.shape() == other.weights.shape()
    }
}

impl Parameterised<Vec<f64>, Vec<f64>> for RBFNetwork
{
    fn update(&mut self, input: &Vec<f64>, errors: Vec<f64>) {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        <Self as QFunction<RegularSpace<Continuous>>>::update_phi(self, &phi, errors);
    }

    fn equivalent(&self, other: &Self) -> bool {
        <Self as Parameterised<Vec<f64>, f64>>::equivalent(self, other)
    }
}


impl Linear<RegularSpace<Continuous>> for RBFNetwork
{
    fn phi(&self, input: &Vec<f64>) -> Array1<f64> {
        let d = &self.mu - &ArrayView::from_shape((1, self.mu.cols()), input).unwrap();
        let e = (&d * &d * &self.gamma).mapv(|v| v.exp()).sum(Axis(1));
        let z = e.sum(Axis(0));

        e / z
    }
}


impl VFunction<RegularSpace<Continuous>> for RBFNetwork
{
    fn evaluate_phi(&self, phi: &Array1<f64>) -> f64 {
        dot(self.weights.column(0).as_slice().unwrap(),
            phi.as_slice().unwrap())
    }

    fn update_phi(&mut self, phi: &Array1<f64>, error: f64) {
        self.weights.column_mut(0).scaled_add(error, phi);
    }
}


impl QFunction<RegularSpace<Continuous>> for RBFNetwork
{
    fn evaluate_action(&self, input: &Vec<f64>, action: usize) -> f64 {
        let phi = self.phi(input);

        self.evaluate_action_phi(&phi, action)
    }

    fn update_action(&mut self, input: &Vec<f64>, action: usize, error: f64) {
        let phi = self.phi(input);

        self.update_action_phi(&phi, action, error);
    }

    fn evaluate_phi(&self, phi: &Array1<f64>) -> Vec<f64> {
        (self.weights.t().dot(phi)).into_raw_vec()
    }

    fn evaluate_action_phi(&self, phi: &Array1<f64>, action: usize) -> f64 {
        self.weights.column(action).dot(phi)
    }

    fn update_phi(&mut self, phi: &Array1<f64>, errors: Vec<f64>) {
        let phi_view = phi.view().into_shape((self.weights.rows(), 1)).unwrap();
        let error_matrix =
            ArrayView::from_shape((1, self.weights.cols()), errors.as_slice()).unwrap();

        self.weights += &phi_view.dot(&error_matrix);
    }

    fn update_action_phi(&mut self, phi: &Array1<f64>, action: usize, error: f64) {
        self.weights.column_mut(action).scaled_add(error, &phi);
    }
}
