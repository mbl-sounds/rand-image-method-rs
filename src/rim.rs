use itertools;
use rand::prelude::*;
use rayon::prelude::*;

pub fn solve_rim(
    h: &mut Vec<Vec<f64>>,
    xs: Vec<f64>,
    xr: Vec<f64>,
    l: Vec<f64>,
    beta: Vec<f64>,
    n: Vec<f64>,
    rd: f64,
    nt: usize,
    seed: u64,
) -> Result<usize, usize> {
    let nr_mics = h.len();

    let l_b = n[0] as i32;
    let m_b = n[1] as i32;
    let n_b = n[2] as i32;

    h.par_iter_mut().enumerate().for_each(|(mic, h_)| {
        let mut rng = StdRng::seed_from_u64(seed);
        for (u_, v_, w_) in itertools::iproduct!(0..=1, 0..=1, 0..=1) {
            for (l_, m_, n_) in itertools::iproduct!(-l_b..l_b, -m_b..m_b, -n_b..n_b) {
                let displacement = if u_ == 0 && v_ == 0 && w_ == 0 && l_ == 0 && m_ == 0 && n_ == 0
                {
                    0.0 // direct path
                } else {
                    rd
                };
                //compute distance
                let pos_x = xs[0] - 2.0 * (u_ as f64) * xs[0]
                    + (l_ as f64) * l[0]
                    + displacement * (2.0 * rng.gen::<f64>() - 1.0);
                let pos_y = xs[1] - 2.0 * (v_ as f64) * xs[1]
                    + (m_ as f64) * l[1]
                    + displacement * (2.0 * rng.gen::<f64>() - 1.0);
                let pos_z = xs[2] - 2.0 * (w_ as f64) * xs[2]
                    + (n_ as f64) * l[2]
                    + displacement * (2.0 * rng.gen::<f64>() - 1.0);

                let b_p = [
                    (l_ - u_).abs(),
                    l_.abs(),
                    (m_ - v_).abs(),
                    m_.abs(),
                    (n_ - w_).abs(),
                    n_.abs(),
                ];

                let d = ((pos_x - xr[0 * nr_mics + mic]).powf(2.0)
                    + (pos_y - xr[1 * nr_mics + mic]).powf(2.0)
                    + (pos_z - xr[2 * nr_mics + mic]).powf(2.0))
                .sqrt();
                let id = d.round() as usize;
                if id >= nt {
                    continue;
                }
                let mut a = b_p
                    .iter()
                    .enumerate()
                    .fold(1.0, |acc, (i, &x)| acc * (beta[i].powf(x as f64)));
                a /= 4.0 * std::f64::consts::PI * d;
                h_[id] += a;
            }
        }
    });

    Ok(0)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     const NT: usize = 16000;

//     #[test]
//     fn solve_rim_test() {
//         let fs = 44100.0;
//         let c = 343.0;
//         let fsc = fs / c;

//         // impulse respones
//         let h_slice: &mut [f64] = &mut [0.0; NT];
//         let h_cell_slice: &Cell<[f64]> = Cell::from_mut(h_slice);
//         let h_mem: &[Cell<f64>] = h_cell_slice.as_slice_of_cells();

//         let h = NdArray {
//             mem: h_mem,
//             dim: 2,
//             shape: &[8000, 2],
//             jumps: vec![2, 1],
//         };

//         // source position
//         let xs_slice: &mut [f64] = &mut [1.0 * fsc, 2.0 * fsc, 3.0 * fsc];
//         let xs_cell_slice: &Cell<[f64]> = Cell::from_mut(xs_slice);
//         let xs_mem: &[Cell<f64>] = xs_cell_slice.as_slice_of_cells();

//         let xs = NdArray {
//             mem: xs_mem,
//             dim: 1,
//             shape: &[3],
//             jumps: vec![1],
//         };

//         // microphone positions
//         let xr_slice: &mut [f64] = &mut [
//             1.0 * fsc,
//             2.0 * fsc,
//             3.0 * fsc,
//             4.0 * fsc,
//             5.0 * fsc,
//             6.0 * fsc,
//         ];
//         let xr_cell_slice: &Cell<[f64]> = Cell::from_mut(xr_slice);
//         let xr_mem: &[Cell<f64>] = xr_cell_slice.as_slice_of_cells();

//         let xr = NdArray {
//             mem: xr_mem,
//             dim: 2,
//             shape: &[3, 2],
//             jumps: vec![2, 1],
//         };

//         // reflection coeffs
//         let beta_slice: &mut [f64] = &mut [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
//         let beta_cell_slice: &Cell<[f64]> = Cell::from_mut(beta_slice);
//         let beta_mem: &[Cell<f64>] = beta_cell_slice.as_slice_of_cells();

//         let beta = NdArray {
//             mem: beta_mem,
//             dim: 1,
//             shape: &[6],
//             jumps: vec![1],
//         };

//         // room dimensions
//         let l_slice: &mut [f64] = &mut [8.0 * fsc * 2.0, 8.0 * fsc * 2.0, 8.0 * fsc * 2.0];
//         let l_cell_slice: &Cell<[f64]> = Cell::from_mut(l_slice);
//         let l_mem: &[Cell<f64>] = l_cell_slice.as_slice_of_cells();

//         let l = NdArray {
//             mem: l_mem,
//             dim: 1,
//             shape: &[3],
//             jumps: vec![1],
//         };

//         // orders
//         let n_slice: &mut [f64] = &mut [1.0, 1.0, 1.0];
//         let n_cell_slice: &Cell<[f64]> = Cell::from_mut(n_slice);
//         let n_mem: &[Cell<f64>] = n_cell_slice.as_slice_of_cells();

//         let n = NdArray {
//             mem: n_mem,
//             dim: 1,
//             shape: &[3],
//             jumps: vec![1],
//         };

//         // displacement
//         let rd = 0.2 * fsc;
//         // number of samples
//         // let nt = 8000;

//         // assert_eq!(Ok(0), solve_rim(h, xs, xr, l, beta, n, rd, NT));
//     }
// }
