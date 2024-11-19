use crate::errors::ArrayCheckError::InvalidArray;
use crate::errors::{ArrayCheckError, UnparallelizableError};
pub use crate::ndiff::interface::DiffInterface;
use ndarray::Array1;
use rayon::prelude::*;

#[allow(dead_code)]
pub struct BackwardDiff {
    array: Array1<f64>,
    use_parallel: bool,
    diff_size: f64,
    order: i32,
}

impl BackwardDiff {
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
        let mut diff_array = Array1::zeros(array_size - 1);
        for i in 1..array_size {
            diff_array[i - 1] = (self.array[i] - self.array[i - 1]) / self.diff_size;
        }
        self.array = diff_array;
        Ok(())
    }

    fn diff_core_first_parallel(&mut self) -> Result<(), ArrayCheckError> {
        let array_size = self.array.len();
        if array_size < 3 {
            return Err(InvalidArray);
        }

        let diff_size = self.diff_size;
        let array_view = self.array.view();
        let mut diff_array = Array1::zeros(array_size - 1);

        // rayon's parallel iterators
        let array_slice = array_view.as_slice().unwrap();
        let diff_slice = diff_array.as_slice_mut().unwrap();

        diff_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = (array_slice[i + 1] - array_slice[i]) / diff_size;
        });

        self.array = diff_array;
        Ok(())
    }

    fn diff_core(&mut self) -> Result<(), ArrayCheckError> {
        match self.order {
            1 => match self.use_parallel {
                true => Ok(self.diff_core_first_parallel()?),
                false => Ok(self.diff_core_first()?),
            },
            _ => panic!("Invalid order number"),
        }
    }
}

impl DiffInterface for BackwardDiff {
    fn parallelize(&self) -> Result<Box<Self>, UnparallelizableError> {
        if self.use_parallel {
            Ok(Box::from(Self {
                array: self.array.clone(),
                use_parallel: true,
                diff_size: self.diff_size.clone(),
                order: self.order,
            }))
        } else {
            if self.array.len() < 3 {
                return Err(UnparallelizableError::Unparallelizable);
            }
            Ok(Box::from(Self {
                array: self.array.clone(),
                use_parallel: false,
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
