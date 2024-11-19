use crate::errors::UnparallelizableError;
use crate::pde::explicit_euler::MeshData::{Mesh2D, Mesh3D};
pub use crate::pde::interface::PDESolverInterface;
use ndarray::{s, Array1, Array2, Array3, Axis};
use std::sync::Arc;

pub struct ExplicitEuler {
    t_sample: i32,
    t_range: Vec<f64>,
    x_sample: i32,
    x_range: Vec<f64>,
    func: Arc<dyn Fn(f64, Array1<f64>) -> Result<Array1<f64>, String>>,
    initial_condition: Array1<f64>,
    spatial_temporal_mesh: MeshData,
    solution_mesh: MeshData,
    use_parallel: bool,
}

pub enum MeshData {
    None,
    Mesh2D(Array2<f64>),
    Mesh3D(Array3<f64>),
}

impl ExplicitEuler {
    #[allow(dead_code)]
    pub fn new(
        t_sample: i32,
        t_range: Vec<f64>,
        x_sample: i32,
        x_range: Vec<f64>,
        func: Arc<dyn Fn(f64, Array1<f64>) -> Result<Array1<f64>, String>>,
        boundary_condition: Array1<f64>,
        use_parallel: bool,
    ) -> Self {
        Self {
            t_sample,
            t_range,
            x_sample,
            x_range,
            func,
            initial_condition: boundary_condition,
            spatial_temporal_mesh: MeshData::None,
            solution_mesh: MeshData::None,
            use_parallel,
        }
    }

    fn generate_spatial_temporal_mesh(&mut self) {
        let ti = Array1::<f64>::linspace(self.t_range[0], self.t_range[1], self.t_sample as usize);
        let xi = Array1::<f64>::linspace(self.x_range[0], self.x_range[1], self.x_sample as usize);

        let mut spatial_temp =
            Array3::<f64>::zeros((self.x_sample as usize, self.t_sample as usize, 2));

        {
            let mut x_slice = spatial_temp.slice_mut(s![.., .., 0]);
            for (i, x) in xi.iter().enumerate() {
                x_slice.row_mut(i).fill(*x);
            }
        }

        {
            let mut t_slice = spatial_temp.slice_mut(s![.., .., 1]);
            for (j, t) in ti.iter().enumerate() {
                t_slice.column_mut(j).fill(*t);
            }
        }

        self.spatial_temporal_mesh = Mesh3D(spatial_temp);
    }

    fn check_function_sanity(&self) -> bool {
        let t_test: f64 = self.t_range[0];
        let x_test =
            Array1::<f64>::linspace(self.x_range[0], self.x_range[1], self.t_sample as usize);

        let x_res = (self.func)(t_test, x_test.clone());
        assert_eq!(
            x_test.shape(),
            x_res.unwrap().shape(),
            "input and output shape of the RHS function are different"
        );
        true
    }

    fn explicit_euler_core(&mut self) {
        let mut solution = Array2::<f64>::zeros((self.x_sample as usize, self.t_sample as usize));
        let mut initial_slice = solution.slice_mut(s![.., 0]);
        initial_slice.assign(&self.initial_condition);

        let mut u_current = self.initial_condition.clone();
        let delta_t = (self.t_range[1] - self.t_range[0]) / (self.t_sample - 1) as f64;

        for (i, mut u_t) in solution
            .slice_mut(s![.., 1..])
            .axis_iter_mut(Axis(1))
            .enumerate()
        {
            let current_t = self.t_range[0] + i as f64 * delta_t;
            let u_next = (self.func)(current_t, u_current.clone()).unwrap() * delta_t + u_current;
            u_t.assign(&u_next);
            u_current = u_next;
        }

        self.solution_mesh = Mesh2D(solution);
    }
}

impl PDESolverInterface for ExplicitEuler {
    fn parallelize(&self) -> Result<Box<Self>, UnparallelizableError> {
        if self.use_parallel {
            Err(UnparallelizableError::Unparallelizable)
        } else {
            Ok(Box::from(Self {
                t_sample: self.t_sample,
                t_range: self.t_range.clone(),
                x_sample: self.x_sample,
                x_range: self.x_range.clone(),
                func: self.func.clone(),
                initial_condition: self.initial_condition.clone(),
                spatial_temporal_mesh: MeshData::None,
                solution_mesh: MeshData::None,
                use_parallel: self.use_parallel.clone(),
            }))
        }
    }

    fn solve(&mut self) -> Result<Array2<f64>, String> {
        self.parallelize().unwrap();
        self.check_function_sanity();
        self.generate_spatial_temporal_mesh();
        self.explicit_euler_core();
        match &self.solution_mesh {
            Mesh3D(solution) => panic!("unexpected solution shape: {:?}", solution),
            Mesh2D(solution) => Ok(solution.clone()),
            MeshData::None => panic!("no solution produced!"),
        }
    }
}
