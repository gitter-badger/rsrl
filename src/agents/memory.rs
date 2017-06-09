use Parameter;
use ndarray::{ArrayBase, Array1};
use std::ops::{AddAssign, MulAssign};


pub enum Trace {
    Accumulating {
        lambda: Parameter,
        eligibility: Array1<f64>,
    },
    Replacing {
        lambda: Parameter,
        eligibility: Array1<f64>,
    },
    // TODO: Dutch traces (need to be able to share alpha parameter)
}

impl Trace {
    pub fn get(&self) -> &Array1<f64> {
        match self {
            &Trace::Accumulating { ref eligibility, .. } |
                &Trace::Replacing { ref eligibility, .. } => eligibility,
        }
    }

    pub fn decay(&mut self, rate: f64) {
        match self {
            &mut Trace::Accumulating { ref mut eligibility, lambda } |
                &mut Trace::Replacing { ref mut eligibility, lambda } =>
            {
                eligibility.mul_assign(rate*lambda);
            },
        }
    }

    pub fn update(&mut self, phi: &Array1<f64>) {
        match self {
            &mut Trace::Accumulating { ref mut eligibility, lambda } =>
            {
                eligibility.add_assign(phi);
            },
            &mut Trace::Replacing { ref mut eligibility, lambda } =>
            {
                eligibility.add_assign(phi);
                eligibility.map_inplace(|val| *val = val.min(1.0));
            },
        }
    }
}


#[cfg(test)]
mod tests {
    use super::Trace;
    use ndarray::{ArrayBase, arr1};

    #[test]
    fn test_accumulating() {
        let mut trace = Trace::Accumulating {
            lambda: 0.95.into(),
            eligibility: arr1(&[0.0f64; 10])
        };

        assert_eq!(trace.get(), &arr1(&[0.0f64; 10]));

        trace.decay(1.0);
        assert_eq!(trace.get(), &arr1(&[0.0f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), &arr1(&[1.0f64; 10]));

        trace.decay(1.0);
        assert_eq!(trace.get(), &arr1(&[0.95f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), &arr1(&[1.95f64; 10]));
    }

    #[test]
    fn test_replacing() {
        let mut trace = Trace::Replacing {
            lambda: 0.95.into(),
            eligibility: arr1(&[0.0f64; 10])
        };

        assert_eq!(trace.get(), &arr1(&[0.0f64; 10]));

        trace.decay(1.0);
        assert_eq!(trace.get(), &arr1(&[0.0f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), &arr1(&[1.0f64; 10]));

        trace.decay(1.0);
        assert_eq!(trace.get(), &arr1(&[0.95f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), &arr1(&[1.0f64; 10]));
    }
}
