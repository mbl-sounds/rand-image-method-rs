use itertools;
use rand::prelude::*;
use rayon::prelude::*;

pub fn solve_rim(
    h: &mut Vec<Vec<f64>>,
    xs: Vec<f64>,
    xr: Vec<f64>,
    l: Vec<f64>,
    beta: Vec<f64>,
    n: Vec<i32>,
    rd: f64,
    nt: usize,
    seed: u64,
) -> Result<usize, usize> {
    let nr_mics = h.len();
    h.par_iter_mut().enumerate().for_each(|(mic, h_)| {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pos = vec![0.0; 3];
        itertools::iproduct!(0..=1, 0..=1, 0..=1).for_each(|coord| {
            itertools::iproduct!(-n[0]..=n[0], -n[1]..=n[1], -n[2]..=n[2]).for_each(|ord| {
                let dis = if coord.eq(&(0, 0, 0)) && ord.eq(&(0, 0, 0)) {
                    0.0 // direct path
                } else {
                    rd
                };

                // mirror source position
                pos[0] = xs[0] - 2.0 * (coord.0 as f64) * xs[0]
                    + (ord.0 as f64) * l[0]
                    + dis * (2.0 * rng.gen::<f64>() - 1.0);
                pos[1] = xs[1] - 2.0 * (coord.1 as f64) * xs[1]
                    + (ord.1 as f64) * l[1]
                    + dis * (2.0 * rng.gen::<f64>() - 1.0);
                pos[2] = xs[2] - 2.0 * (coord.2 as f64) * xs[2]
                    + (ord.2 as f64) * l[2]
                    + dis * (2.0 * rng.gen::<f64>() - 1.0);

                // dampening power
                let b_p = [
                    (ord.0 - coord.0).abs(),
                    ord.0,
                    (ord.1 - coord.1).abs(),
                    ord.1,
                    (ord.2 - coord.2).abs(),
                    ord.2,
                ];

                // distance to mirror source
                let d = pos
                    .iter()
                    .enumerate()
                    .fold(0.0, |acc, (i, &x)| {
                        acc + (x - xr[i * nr_mics + mic]).powf(2.0)
                    })
                    .sqrt();

                let id = d.round() as usize;
                if id < nt {
                    // amplitude
                    h_[id] += beta
                        .iter()
                        .enumerate()
                        .fold(1.0, |acc, (i, &b)| acc * (b.powf(b_p[i] as f64)))
                        / (4.0 * std::f64::consts::PI * d);
                }
            });
        });
    });

    Ok(0)
}

fn add_delay(h: &mut Vec<f64>, d: f64, a: f64, tw: f64, fc: f64, nt: usize) -> () {
    let start_idx = (d - tw / 2.0).ceil().max(0.0) as usize;
    let end_idx = (d + tw / 2.0).floor().min(nt as f64) as usize;
    let a2 = a / 2.0;
    h[start_idx..end_idx]
        .iter_mut()
        .enumerate()
        .for_each(|(i, val)| {
            *val += a2
                * (1.0 + (2.0 * std::f64::consts::PI * (i as f64 - d) / tw).cos())
                * sinc(fc * (i as f64 - d));
        });
}

fn sinc(x: f64) -> f64 {
    (x * std::f64::consts::PI).sin() / (x * std::f64::consts::PI)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn solve_rim_test() {
//         // assert_eq!(Ok(0), solve_rim(h, xs, xr, l, beta, n, rd, NT));
//     }
// }
