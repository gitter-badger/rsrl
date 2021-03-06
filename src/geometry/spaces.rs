use super::Span;
use super::dimensions;
use super::dimensions::{Dimension, Partitioned};
use rand::ThreadRng;
use std::iter::FromIterator;
use std::slice::Iter;


pub trait Space {
    type Repr: Clone;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr;

    fn dim(&self) -> usize;
    fn span(&self) -> Span;
}

pub type ActionSpace = UnitarySpace<dimensions::Discrete>;


#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct NullSpace;

impl Space for NullSpace {
    type Repr = ();

    fn sample(&self, _: &mut ThreadRng) -> Self::Repr {
        ()
    }

    fn dim(&self) -> usize {
        0
    }

    fn span(&self) -> Span {
        Span::Null
    }
}


#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct UnitarySpace<D: Dimension>(D);

impl<D: Dimension> UnitarySpace<D> {
    pub fn new(d: D) -> Self {
        UnitarySpace(d)
    }
}

impl<D: Dimension> Space for UnitarySpace<D> {
    type Repr = D::Value;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr {
        self.0.sample(rng)
    }

    fn dim(&self) -> usize {
        1
    }

    fn span(&self) -> Span {
        self.0.span()
    }
}


#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct PairSpace<D1, D2>((D1, D2))
    where D1: Dimension,
          D2: Dimension;

impl<D1: Dimension, D2: Dimension> PairSpace<D1, D2> {
    pub fn new(d1: D1, d2: D2) -> Self {
        PairSpace((d1, d2))
    }
}

impl<D1: Dimension, D2: Dimension> Space for PairSpace<D1, D2> {
    type Repr = (D1::Value, D2::Value);

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr {
        ((self.0).0.sample(rng), (self.0).1.sample(rng))
    }

    fn dim(&self) -> usize {
        2
    }

    fn span(&self) -> Span {
        (self.0).0.span()*(self.0).1.span()
    }
}

impl PairSpace<dimensions::Continuous, dimensions::Continuous> {
    pub fn partitioned(self, density: usize) -> PairSpace<Partitioned, Partitioned> {
        PairSpace((Partitioned::from_continuous((self.0).0, density),
                   Partitioned::from_continuous((self.0).1, density)))
    }
}


#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RegularSpace<D: Dimension> {
    dimensions: Vec<D>,
    span: Span,
}

impl<D: Dimension> RegularSpace<D> {
    pub fn new() -> Self {
        RegularSpace {
            dimensions: vec![],
            span: Span::Null,
        }
    }

    pub fn push(mut self, d: D) -> Self {
        self.span = self.span * d.span();
        self.dimensions.push(d);
        self
    }

    pub fn iter(&self) -> Iter<D> {
        self.dimensions.iter()
    }
}

impl<D: Dimension> Space for RegularSpace<D> {
    type Repr = Vec<D::Value>;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr {
        self.dimensions.iter().map(|d| d.sample(rng)).collect()
    }

    fn dim(&self) -> usize {
        self.dimensions.len()
    }

    fn span(&self) -> Span {
        self.span
    }
}

impl RegularSpace<dimensions::Continuous> {
    pub fn partitioned(self, density: usize) -> RegularSpace<Partitioned> {
        self.into_iter()
            .map(|d| Partitioned::from_continuous(d, density))
            .collect()
    }
}

impl RegularSpace<dimensions::Partitioned> {
    pub fn centres(&self) -> Vec<Vec<f64>> {
        self.dimensions
            .iter()
            .map(|d| d.centres())
            .collect()
    }
}

impl<D: Dimension> FromIterator<D> for RegularSpace<D> {
    fn from_iter<I: IntoIterator<Item = D>>(iter: I) -> Self {
        let mut s = Self::new();

        for i in iter {
            s = s.push(i);
        }

        s
    }
}

impl<D: Dimension> IntoIterator for RegularSpace<D> {
    type Item = D;
    type IntoIter = ::std::vec::IntoIter<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.dimensions.into_iter()
    }
}


// pub struct HeterogeneousSpace {
// dimensions: Vec<Dimension>,
// span: Span
// }

// impl HeterogeneousSpace {
// pub fn new() -> Self {
// HeterogeneousSpace {
// dimensions: vec![],
// span: Span::Null
// }
// }

// pub fn push(mut self, d: D) -> Self {
// self.span = self.span * d.span();
// self.dimensions.push(d);
// self
// }

// pub fn iter(&self) -> Iter<Dimension> {
// self.dimensions.iter()
// }
// }

// impl<D: Dimension> Space for HeterogeneousSpace<D> {
// type Repr = Vec<D::Value>;

// fn sample(&self, rng: &mut ThreadRng) -> Self::Repr {
// self.dimensions.iter().map(|d| d.sample(rng)).collect()
// }

// fn dim(&self) -> usize {
// self.dimensions.len()
// }

// fn span(&self) -> Span {
// self.span
// }
// }


#[cfg(test)]
mod tests {
    use super::{Space, NullSpace, UnitarySpace, PairSpace, RegularSpace};
    use geometry::Span;
    use geometry::dimensions::*;
    use ndarray::arr1;
    use rand::thread_rng;

    #[test]
    fn test_null_space() {
        let ns = NullSpace;
        let mut rng = thread_rng();

        assert_eq!(ns.sample(&mut rng), ());
        assert_eq!(ns.dim(), 0);
        assert_eq!(ns.span(), Span::Null);
    }

    #[test]
    fn test_unitary_space() {
        let d = Discrete::new(2);
        let us = UnitarySpace::new(d);
        let mut rng = thread_rng();

        let mut counts = arr1(&vec![0.0; 2]);
        for _ in 0..5000 {
            let sample = us.sample(&mut rng);
            counts[sample] += 1.0;

            assert!(sample == 0 || sample == 1);
        }

        assert!((counts/5000.0).all_close(&arr1(&vec![0.5; 2]), 1e-1));
        assert_eq!(us.dim(), 1);
        assert_eq!(us.span(), d.span());
    }

    #[test]
    fn test_pair_space() {
        let d1 = Discrete::new(2);
        let d2 = Discrete::new(2);
        let ps = PairSpace::new(d1, d2);

        let mut rng = thread_rng();

        let mut c1 = arr1(&vec![0.0; 2]);
        let mut c2 = arr1(&vec![0.0; 2]);
        for _ in 0..5000 {
            let sample = ps.sample(&mut rng);

            c1[sample.0] += 1.0;
            c2[sample.1] += 1.0;

            assert!(sample.0 == 0 || sample.0 == 1);
            assert!(sample.1 == 0 || sample.1 == 1);
        }

        assert!((c1/5000.0).all_close(&arr1(&vec![0.5; 2]), 1e-1));
        assert!((c2/5000.0).all_close(&arr1(&vec![0.5; 2]), 1e-1));

        assert_eq!(ps.dim(), 2);
        assert_eq!(ps.span(), d1.span()*d2.span());
    }
}
