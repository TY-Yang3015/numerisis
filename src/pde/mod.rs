mod explicit_euler;
mod interface;

pub mod pde {

    #[cfg(test)]
    mod tests {
        use crate::ndiff::central::{CentralDiff, DiffInterface};
        use crate::pde::explicit_euler::*;
        use ndarray::{concatenate, Array1, Axis};
        use plotters::prelude::*;
        use std::sync::Arc;

        #[test]
        fn test_explicit_euler_ode() {
            // PDE can be reduced to ODE by dropping spatial dependence

            let t_sample = 1000; // number of time steps
            let t_range = vec![0.0, 1.0]; // time range [start, end]
            let x_sample = 1; // number of spatial points, 1 meaning independent
                              // of space coordinate
            let x_range = vec![0.0, 1.0]; // spatial range [start, end]

            // define the function (RHS of the ODE)
            // du/dt = -u
            let func = Arc::new(|_t: f64, u: Array1<f64>| -> Result<Array1<f64>, String> {
                Ok(-u.mapv(|val| val))
            });

            // u(t=0) = 1
            let initial_condition = Array1::linspace(1.0, 1.0, x_sample as usize);

            // initialize the ExplicitEuler solver
            let mut solver = ExplicitEuler::new(
                t_sample,
                t_range.clone(),
                x_sample,
                x_range,
                func,
                initial_condition.clone(),
                false,
                None
            );

            // solve the ODE
            match solver.solve() {
                Ok(solution) => {
                    println!("ODE solution shape: {:?}", solution.shape());

                    // last element should be close to 0.3677
                    println!("solution:");
                    println!("{:?}", solution);

                    let time_steps =
                        Array1::<f64>::linspace(t_range[0], t_range[1], t_sample as usize);
                    let solution_values = solution.row(0).to_vec();

                    let root = BitMapBackend::new("explicit_euler_ode_result.png", (1920, 1080))
                        .into_drawing_area();
                    root.fill(&WHITE).unwrap();

                    let mut chart = ChartBuilder::on(&root)
                        .caption("Explicit Euler ODE Solution", ("sans-serif", 50))
                        .margin(20)
                        .x_label_area_size(40)
                        .y_label_area_size(40)
                        .build_cartesian_2d(t_range[0]..t_range[1], 0.0..1.0)
                        .unwrap();

                    chart
                        .configure_mesh()
                        .x_label_style(("sans-serif", 40)) // Set x-axis label font size
                        .y_label_style(("sans-serif", 40))
                        .draw()
                        .unwrap();

                    chart
                        .draw_series(LineSeries::new(
                            time_steps
                                .iter()
                                .zip(solution_values.iter())
                                .map(|(&t, &u)| (t, u)),
                            &RED,
                        ))
                        .unwrap()
                        .label("u(t)")
                        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

                    chart
                        .configure_series_labels()
                        .background_style(&WHITE.mix(0.8))
                        .border_style(&BLACK)
                        .label_font(("sans-serif", 40))
                        .draw()
                        .unwrap();

                    println!("Plot saved as 'explicit_euler_ode_result.png'");
                }
                Err(err) => {
                    println!("Failed to compute solution: {}", err);
                }
            }
        }

        #[test]
        fn test_explicit_euler_pde() {
            // solve diffusion PDE

            let t_sample: i32 = 100000; // number of time steps
            let t_range = vec![0.0, 0.8]; // time range [start, end]
            let x_sample = 1000; // number of spatial points, 1 meaning independent
                                 // of space coordinate
            let x_range = vec![0.0, 10.0]; // spatial range [start, end]
            let delta_x = (x_range[1] - x_range[0]) / ((x_sample - 1) as f64);

            // test boundary condition, l = 1, r = 0
            let r_boundary = Array1::zeros((t_sample - 1) as usize);
            let l_boundary = Array1::ones((t_sample - 1) as usize);

            // define the function (RHS of the ODE)
            // du/dt = D d^2u/dx^2, D = 1
            // initial condition is a triangular pulse
            let func = Arc::new(
                move |_t: f64, u: Array1<f64>| -> Result<Array1<f64>, String> {
                    let mut central_diff = CentralDiff::new(u.clone(), false, delta_x, 2);
                    let diff_result = central_diff.differentiate();

                    // handle differentiation errors
                    let diff = match diff_result {
                        Ok(d) => d,
                        Err(e) => return Err(format!("Differentiation failed: {:?}", e)),
                    };

                    // add zeros at the start and end of the array
                    let padded_diff = concatenate(
                        Axis(0),
                        &[
                            Array1::from_elem(1, 0.).view(),
                            diff.view(),
                            Array1::from_elem(1, 0.).view(),
                        ],
                    )
                    .map_err(|e| format!("Stacking failed: {}", e))?;

                    // flatten the 2D array back to a 1D array
                    let result = padded_diff
                        .clone()
                        .into_shape_with_order((padded_diff.len(),))
                        .map_err(|e| format!("Reshaping failed: {}", e))?;

                    Ok(result)
                },
            );

            fn triangular_function(size: usize, start: f64, peak: f64, end: f64) -> Array1<f64> {
                let mut arr = Array1::zeros(size);

                // calculate the normalization factor
                let normalization_factor = 1.0 / ((peak - start).max(end - peak));

                for (i, x) in arr.iter_mut().enumerate() {
                    let t = i as f64 / (size as f64 - 1.0);

                    if t >= start && t < peak {
                        *x = ((t - start) / (peak - start)) * normalization_factor;
                    } else if t >= peak && t <= end {
                        *x = ((end - t) / (end - peak)) * normalization_factor;
                    } else {
                        *x = 0.0;
                    }
                }
                arr * 0.5
            }
            let initial_condition = triangular_function(x_sample as usize, 0.1, 0.2, 0.3);
            println!("initial condition: {:?}", initial_condition);

            // initialize the ExplicitEuler solver
            let mut solver = ExplicitEuler::new(
                t_sample,
                t_range.clone(),
                x_sample.try_into().unwrap(),
                x_range.clone(),
                func,
                initial_condition.clone(),
                false,
                Some((l_boundary, r_boundary))
            );

            // solve the PDE
            match solver.solve() {
                Ok(solution) => {
                    println!("PDE solution shape: {:?}", solution.shape());

                    let x_steps =
                        Array1::<f64>::linspace(x_range[0], x_range[1], x_sample as usize);
                    let solution_values = solution.column(solution.shape()[1] - 1).to_vec();
                    let init_values = solution.column(0).to_vec();

                    let root = BitMapBackend::new("explicit_euler_pde_result.png", (1920, 1080))
                        .into_drawing_area();
                    root.fill(&WHITE).unwrap();

                    let mut chart = ChartBuilder::on(&root)
                        .caption("Explicit Euler ODE Solution", ("sans-serif", 50))
                        .margin(20)
                        .x_label_area_size(40)
                        .y_label_area_size(40)
                        .build_cartesian_2d(x_range[0]..x_range[1], 0.0..5.)
                        .unwrap();

                    chart
                        .configure_mesh()
                        .x_label_style(("sans-serif", 40)) // Set x-axis label font size
                        .y_label_style(("sans-serif", 40))
                        .draw()
                        .unwrap();

                    chart
                        .draw_series(LineSeries::new(
                            x_steps
                                .iter()
                                .zip(solution_values.iter())
                                .map(|(&t, &u)| (t, u)),
                            &RED,
                        ))
                        .unwrap()
                        .label("u(t_f)")
                        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

                    chart
                        .draw_series(LineSeries::new(
                            x_steps
                                .iter()
                                .zip(init_values.iter())
                                .map(|(&t, &u)| (t, u)),
                            &BLUE,
                        ))
                        .unwrap()
                        .label("u(t_0)")
                        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

                    chart
                        .configure_series_labels()
                        .background_style(&WHITE.mix(0.8))
                        .border_style(&BLACK)
                        .label_font(("sans-serif", 40))
                        .draw()
                        .unwrap();

                    println!("Plot saved as 'explicit_euler_pde_result.png'");
                }
                Err(err) => {
                    println!("Failed to compute solution: {}", err);
                }
            }
        }
    }
}
