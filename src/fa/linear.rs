use super::{Function, Parameterised, VFunction, QFunction, Projection, SparseProjection};
use geometry::Space;
use ndarray::{ArrayView, Array1, Array2};
use std::marker::PhantomData;
use utils::dot;


pub type Linear<S, P> = DenseLinear<S, P>;


#[derive(Serialize, Deserialize)]
pub struct DenseLinear<S: Space, P: Projection<S>> {
    pub projector: P,
    pub weights: Array2<f64>,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Projection<S>> DenseLinear<S, P> {
    pub fn new(projector: P, n_outputs: usize) -> Self {
        let n_features = projector.size();

        Self {
            projector: projector,
            weights: Array2::<f64>::zeros((n_features, n_outputs)),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Projection<S>> Function<S::Repr, f64> for DenseLinear<S, P> {
    fn evaluate(&self, input: &S::Repr) -> f64 {
        // Compute the feature vector phi:
        let phi = self.projector.project(input);

        <Self as VFunction<S>>::evaluate_phi(self, &phi)
    }
}

impl<S: Space, P: Projection<S>> Function<S::Repr, Vec<f64>> for DenseLinear<S, P> {
    fn evaluate(&self, input: &S::Repr) -> Vec<f64> {
        // Compute the feature vector phi:
        let phi = self.projector.project(input);

        // Apply matrix multiplication and return Vec<f64>:
        <Self as QFunction<S>>::evaluate_phi(self, &phi)
    }
}

impl<S: Space, P: Projection<S>> Parameterised<S::Repr, f64> for DenseLinear<S, P> {
    fn update(&mut self, input: &S::Repr, error: f64) {
        // Compute the feature vector phi:
        let phi = self.projector.project(input);

        <Self as VFunction<S>>::update_phi(self, &phi, error);
    }
}

impl<S: Space, P: Projection<S>> Parameterised<S::Repr, Vec<f64>> for DenseLinear<S, P> {
    fn update(&mut self, input: &S::Repr, errors: Vec<f64>) {
        // Compute the feature vector phi:
        let phi = self.projector.project(input);

        <Self as QFunction<S>>::update_phi(self, &phi, errors);
    }
}

impl<S: Space, P: Projection<S>> VFunction<S> for DenseLinear<S, P> {
    fn evaluate_phi(&self, phi: &Array1<f64>) -> f64 {
        dot(self.weights.column(0).as_slice().unwrap(),
            phi.as_slice().unwrap())
    }

    fn update_phi(&mut self, phi: &Array1<f64>, error: f64) {
        self.weights.column_mut(0).scaled_add(error, phi);
    }
}

impl<S: Space, P: Projection<S>> QFunction<S> for DenseLinear<S, P> {
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64 {
        let phi = self.projector.project(input);

        self.evaluate_action_phi(&phi, action)
    }

    fn update_action(&mut self, input: &S::Repr, action: usize, error: f64) {
        let phi = self.projector.project(input);

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
        let error_matrix = ArrayView::from_shape((1, self.weights.cols()), errors.as_slice())
            .unwrap();

        self.weights += &phi_view.dot(&error_matrix);
    }

    fn update_action_phi(&mut self, phi: &Array1<f64>, action: usize, error: f64) {
        self.weights.column_mut(action).scaled_add(error, &phi);
    }
}

impl<S: Space, P: Projection<S>> Projection<S> for DenseLinear<S, P> {
    fn project(&self, input: &S::Repr) -> Array1<f64> {
        self.projector.project(input)
    }

    fn project_onto(&self, input: &S::Repr, phi: &mut Array1<f64>) {
        self.projector.project_onto(input, phi);
    }

    fn dim(&self) -> usize {
        self.projector.dim()
    }

    fn size(&self) -> usize {
        self.projector.size()
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.projector.equivalent(&other.projector)
    }
}


#[derive(Serialize, Deserialize)]
pub struct SparseLinear<S: Space, P: SparseProjection<S>> {
    pub projector: P,
    pub weights: Array2<f64>,

    phantom: PhantomData<S>,
}

impl<S: Space, P: SparseProjection<S>> SparseLinear<S, P> {
    pub fn new(projector: P, n_outputs: usize) -> Self {
        let n_features = projector.size();

        Self {
            projector: projector,
            weights: Array2::<f64>::zeros((n_features, n_outputs)),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: SparseProjection<S>> Function<S::Repr, f64> for SparseLinear<S, P> {
    fn evaluate(&self, input: &S::Repr) -> f64 {
        self.projector.project_sparse(input).fold(0.0f64,
                                                  |acc, i| acc + self.weights.column(0)[*i])
    }
}

impl<S: Space, P: SparseProjection<S>> Function<S::Repr, Vec<f64>> for SparseLinear<S, P> {
    fn evaluate(&self, input: &S::Repr) -> Vec<f64> {
        let indices = self.projector.project_sparse(input);

        (0..self.weights.cols()).map(|c| {
            indices.fold(0.0f64, |acc, i| acc + self.weights.column(c)[*i])
        }).collect()
    }
}

impl<S: Space, P: SparseProjection<S>> Parameterised<S::Repr, f64> for SparseLinear<S, P> {
    fn update(&mut self, input: &S::Repr, error: f64) {
        let scaled_error = error / self.projector.sparsity() as f64;

        for i in self.projector.project_sparse(input).iter() {
            self.weights.column_mut(0)[*i] += scaled_error;
        }
    }
}

impl<S: Space, P: SparseProjection<S>> Parameterised<S::Repr, Vec<f64>> for SparseLinear<S, P> {
    fn update(&mut self, input: &S::Repr, errors: Vec<f64>) {
        for c in 0..self.weights.cols() {
            let scaled_error = errors[c] / self.projector.sparsity() as f64;

            for i in self.projector.project_sparse(input).iter() {
                self.weights.column_mut(c)[*i] += scaled_error;
            }
        }
    }
}

impl<S: Space, P: SparseProjection<S>> VFunction<S> for SparseLinear<S, P> {}

impl<S: Space, P: SparseProjection<S>> QFunction<S> for SparseLinear<S, P> {
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64 {
        self.projector.project_sparse(input).fold(0.0f64,
                                                  |acc, i| acc + self.weights.column(action)[*i])
    }

    fn update_action(&mut self, input: &S::Repr, action: usize, error: f64) {
        let scaled_error = error / self.projector.sparsity() as f64;

        for i in self.projector.project_sparse(input).iter() {
            self.weights.column_mut(action)[*i] += scaled_error;
        }
    }
}
