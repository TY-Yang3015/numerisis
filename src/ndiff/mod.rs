pub mod backward;
pub mod central;
mod interface;

pub mod diff {

    #[cfg(test)]
    mod tests {
        use crate::ndiff::backward::*;
        use crate::ndiff::central::*;
        use ndarray::{s, Array1};
        use std::ops::{Mul, Sub};

        #[test]
        fn test_central_diff_1() {
            let test_array = Array1::range(0., 5000., 1.);
            let square_array: Array1<f64> = test_array.clone().mapv(|x| x * x);

            let mut central_diff = CentralDiff::new(square_array, false, 1., 1);

            let ansatz = test_array.slice(s![1..4999]).mul(2.);

            match central_diff.differentiate() {
                Ok(result) => {
                    assert_eq!(result, ansatz);
                }
                Err(err) => {
                    println!("Error during differentiation: {:?}", err);
                }
            }
        }

        #[test]
        fn test_central_diff_2() {
            let test_array = Array1::range(0., 5000., 1.);
            let square_array: Array1<f64> = test_array.clone().mapv(|x| x * x);

            let mut central_diff = CentralDiff::new(square_array, false, 1., 2);

            let ansatz = Array1::<f64>::ones(test_array.len() - 2).mul(2.);

            match central_diff.differentiate() {
                Ok(result) => {
                    assert_eq!(result, ansatz);
                }
                Err(err) => {
                    println!("Error during differentiation: {:?}", err);
                }
            }
        }

        #[test]
        fn test_backward_diff_1() {
            let test_array = Array1::range(0., 5000., 1.);
            let square_array: Array1<f64> = test_array.clone().mapv(|x| x * x);

            let mut backward_diff = BackwardDiff::new(square_array, false, 1., 1);

            let ansatz = test_array.slice(s![1..5000]).sub(0.5).mul(2.);

            match backward_diff.differentiate() {
                Ok(result) => {
                    assert_eq!(result, ansatz);
                }
                Err(err) => {
                    println!("Error during differentiation: {:?}", err);
                }
            }
        }

        #[test]
        fn test_backward_diff_1_parallel() {
            let test_array = Array1::range(0., 5000., 1.);
            let square_array: Array1<f64> = test_array.clone().mapv(|x| x * x);

            let mut backward_diff = BackwardDiff::new(square_array, true, 1., 1);

            let ansatz = test_array.slice(s![1..5000]).sub(0.5).mul(2.);

            match backward_diff.differentiate() {
                Ok(result) => {
                    assert_eq!(result, ansatz);
                }
                Err(err) => {
                    println!("Error during differentiation: {:?}", err);
                }
            }
        }
    }
}
