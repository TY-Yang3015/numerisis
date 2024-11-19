use crate::errors::{ArrayCheckError, UnparallelizableError};
use ndarray;
use ndarray::Array1;

pub trait DiffInterface {
    fn parallelize(&self) -> Result<Box<Self>, UnparallelizableError>;
    #[allow(dead_code)]
    fn differentiate(&mut self) -> Result<Array1<f64>, ArrayCheckError>;
}
