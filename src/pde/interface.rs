use crate::errors::UnparallelizableError;
use ndarray::Array2;

pub trait PDESolverInterface {
    fn parallelize(&self) -> Result<Box<Self>, UnparallelizableError>;
    #[allow(dead_code)]
    fn solve(&mut self) -> Result<Array2<f64>, String>;
}
