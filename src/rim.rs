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
    tw: usize,
    fc: f64,
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
                    ord.0.abs(),
                    (ord.1 - coord.1).abs(),
                    ord.1.abs(),
                    (ord.2 - coord.2).abs(),
                    ord.2.abs(),
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
                    if tw == 0 {
                        h_[id] += get_amplitude(&beta, &b_p, d);
                    } else {
                        add_delay(h_, d, get_amplitude(&beta, &b_p, d), tw as f64, fc, nt);
                    }
                }
            });
        });
    });

    Ok(0)
}

fn get_amplitude(beta: &Vec<f64>, b_p: &[i32], d: f64) -> f64 {
    beta.iter()
        .enumerate()
        .fold(1.0, |acc, (i, &b)| acc * (b.powf(b_p[i] as f64)))
        / (4.0 * std::f64::consts::PI * d)
}

fn add_delay(h: &mut Vec<f64>, d: f64, a: f64, tw: f64, fc: f64, nt: usize) -> () {
    let start_idx = (d - tw / 2.0).ceil().max(0.0) as usize;
    let end_idx = (d + tw / 2.0).floor().min(nt as f64) as usize;
    let a2 = a / 2.0;
    h[start_idx..end_idx]
        .iter_mut()
        .enumerate()
        .for_each(|(i, val)| {
            let ind = (i + start_idx) as f64;
            *val += a2
                * (1.0 + (2.0 * std::f64::consts::PI * (ind - d) / tw).cos())
                * sinc(fc * (ind - d));
        });
}

fn sinc(x: f64) -> f64 {
    (x * std::f64::consts::PI).sin() / (x * std::f64::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    fn mult_vec(vec: &mut Vec<f64>, mul: f64) -> () {
        vec.iter_mut().for_each(|val| *val *= mul);
    }
    #[test]
    fn mic3_rd_tw() {
        let fs = 44100.0;
        let c = 343.0;
        let fsc = fs / c;

        let nt = 44100;
        let mut h: Vec<Vec<f64>> = Vec::new();
        for _ in 0..3 {
            h.push(vec![0.0; nt]);
        }
        let mut xs = vec![1.0, 1.0, 1.0];
        mult_vec(&mut xs, fsc);
        let mut xr = vec![1.0, 3.0, 5.1, 2.0, 4.0, 4.7, 3.0, 5.0, 5.0];
        mult_vec(&mut xr, fsc);
        let mut l = vec![6.0, 6.0, 6.0];
        mult_vec(&mut l, 2.0 * fsc);
        let beta = vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
        let n = vec![10, 10, 10];
        let rd = 0.2 * fsc;
        let tw: usize = 20;
        let fc = 0.9;
        let seed: u64 = 1234567;
        let start = Instant::now();
        let res = solve_rim(&mut h, xs, xr, l, beta, n, rd, nt, tw, fc, seed);
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed);
        assert_eq!(Ok(0), res);
    }
    #[test]
    fn mic3_tw() {
        let fs = 44100.0;
        let c = 343.0;
        let fsc = fs / c;

        let nt = 44100;
        let mut h: Vec<Vec<f64>> = Vec::new();
        for _ in 0..3 {
            h.push(vec![0.0; nt]);
        }
        let mut xs = vec![1.0, 1.0, 1.0];
        mult_vec(&mut xs, fsc);
        let mut xr = vec![1.0, 3.0, 5.1, 2.0, 4.0, 4.7, 3.0, 5.0, 5.0];
        mult_vec(&mut xr, fsc);
        let mut l = vec![6.0, 6.0, 6.0];
        mult_vec(&mut l, 2.0 * fsc);
        let beta = vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
        let n = vec![20, 20, 20];
        let rd = 0.0 * fsc;
        let tw: usize = 20;
        let fc = 0.9;
        let seed: u64 = 1234567;
        let start = Instant::now();
        let res = solve_rim(&mut h, xs, xr, l, beta, n, rd, nt, tw, fc, seed);
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed);
        assert_eq!(Ok(0), res);
    }
}
