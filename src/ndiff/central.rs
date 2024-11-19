use crate::errors::ArrayCheckError::InvalidArray;
use crate::errors::{ArrayCheckError, UnparallelizableError};
pub use crate::ndiff::interface::DiffInterface;
use ndarray::Array1;

#[allow(dead_code)]
pub struct CentralDiff {
    array: Array1<f64>,
    use_parallel: bool,
    diff_size: f64,
    order: i32,
}

impl CentralDiff {
    #[allow(dead_code)]
    pub fn new(array: Array1<f64>, use_parallel: bool, diff_size: f64, order: i32) -> Self {
        Self {
            array,
            use_parallel,
            diff_size,
            order,
        }
    }

    fn diff_core_first(&mut self) -> Result<(), ArrayCheckError> {
        let array_size = self.array.len();
        if array_size < 3 {
            return Err(InvalidArray);
        }
        let mut diff_array = Array1::zeros(array_size - 2);
        for i in 1..(array_size - 1) {
            diff_array[i - 1] = (self.array[i + 1] - self.array[i - 1]) / (2. * self.diff_size);
        }
        self.array = diff_array;
        Ok(())
    }

    fn diff_core_second(&mut self) -> Result<(), ArrayCheckError> {
        let array_size = self.array.len();
        if array_size < 3 {
            return Err(InvalidArray);
        }
        let mut diff_array = Array1::zeros(array_size - 2);
        for i in 1..(array_size - 1) {
            diff_array[i - 1] = (self.array[i + 1] + self.array[i - 1] - 2. * self.array[i])
                / (self.diff_size * self.diff_size);
        }
        self.array = diff_array;
        Ok(())
    }

    fn diff_core(&mut self) -> Result<(), ArrayCheckError> {
        match self.order {
            1 => self.diff_core_first(),
            2 => self.diff_core_second(),
            _ => panic!("Invalid order number"),
        }
    }
}

impl DiffInterface for CentralDiff {
    fn parallelize(&self) -> Result<Box<Self>, UnparallelizableError> {
        if self.use_parallel {
            Err(UnparallelizableError::Unparallelizable)
        } else {
            Ok(Box::from(Self {
                array: self.array.clone(),
                use_parallel: self.use_parallel,
                diff_size: self.diff_size.clone(),
                order: self.order,
            }))
        }
    }

    fn differentiate(&mut self) -> Result<Array1<f64>, ArrayCheckError> {
        self.parallelize().unwrap();
        self.diff_core()?;
        Ok(self.array.clone())
    }
}
