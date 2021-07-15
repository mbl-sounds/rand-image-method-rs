mod rim;
use cpython::*;

// add bindings to the generated python module
py_module_initializer!(librimrs, |py, m| {
    m.add(
        py,
        "rim",
        py_fn!(
            py,
            solve_rim_py(
                h_py: PyObject,
                xs_py: PyObject,
                xr_py: PyObject,
                l_py: PyObject,
                beta_py: PyObject,
                n_py: PyObject,
                rd: f64,
                nt: usize,
                tw: usize,
                fc: f64,
                seed: u64,
                xr_dir_py: PyObject,
                xr_type_py: PyObject
            )
        ),
    )?;
    Ok(())
});

fn solve_rim_py(
    py: Python,
    h_py: PyObject,
    xs_py: PyObject,
    xr_py: PyObject,
    l_py: PyObject,
    beta_py: PyObject,
    n_py: PyObject,
    rd: f64,
    nt: usize,
    tw: usize,
    fc: f64,
    seed: u64,
    xr_dir_py: PyObject,
    xr_type_py: PyObject,
) -> PyResult<usize> {
    let h_buf = match buffer::PyBuffer::get(py, &h_py) {
        Ok(buffer) => buffer,
        Err(err) => return Err(err),
    };
    let nr_mics = h_buf.shape()[1];
    let mut h: Vec<Vec<f64>> = Vec::new();
    for _ in 0..nr_mics {
        h.push(vec![0.0; nt]);
    }

    let xs = match buffer::PyBuffer::get(py, &xs_py) {
        Ok(buffer) => buffer.to_vec::<f64>(py).unwrap(),
        Err(err) => return Err(err),
    };

    let xr_ = match buffer::PyBuffer::get(py, &xr_py) {
        Ok(buffer) => buffer.to_vec::<f64>(py).unwrap(),
        Err(err) => return Err(err),
    };
    let mut xr: Vec<Vec<f64>> = Vec::new();
    for mic in 0..nr_mics {
        let mut pos = Vec::new();
        for i in 0..3 {
            pos.push(xr_[i * nr_mics + mic]);
        }
        xr.push(pos);
    }

    let l = match buffer::PyBuffer::get(py, &l_py) {
        Ok(buffer) => buffer.to_vec::<f64>(py).unwrap(),
        Err(err) => return Err(err),
    };

    let beta = match buffer::PyBuffer::get(py, &beta_py) {
        Ok(buffer) => buffer.to_vec::<f64>(py).unwrap(),
        Err(err) => return Err(err),
    };

    let n = match buffer::PyBuffer::get(py, &n_py) {
        Ok(buffer) => buffer.to_vec::<i32>(py).unwrap(),
        Err(err) => return Err(err),
    };

    let xr_dir_ = match buffer::PyBuffer::get(py, &xr_dir_py) {
        Ok(buffer) => buffer.to_vec::<f64>(py).unwrap(),
        Err(err) => return Err(err),
    };
    let mut xr_dir: Vec<Vec<f64>> = Vec::new();
    for mic in 0..nr_mics {
        let mut pos = Vec::new();
        for i in 0..3 {
            pos.push(xr_dir_[i * nr_mics + mic]);
        }
        xr_dir.push(pos);
    }

    let xr_type_ = match buffer::PyBuffer::get(py, &xr_type_py) {
        Ok(buffer) => buffer.to_vec::<u8>(py).unwrap(),
        Err(err) => return Err(err),
    };
    let xr_type = xr_type_
        .iter()
        .map(|&val| match val {
            1 => 'b',
            2 => 'h',
            3 => 'c',
            4 => 's',
            _ => 'o',
        })
        .collect::<Vec<char>>();

    match rim::solve_rim(
        &mut h, xs, xr, l, beta, n, rd, nt, tw, fc, seed, xr_dir, xr_type,
    ) {
        Ok(res) => {
            let nr_mics = h.len();
            let slice = h_buf.as_mut_slice::<f64>(py).unwrap();
            h.iter().enumerate().for_each(|(mic, h_)| {
                h_.iter()
                    .enumerate()
                    .for_each(|(i, &val)| slice[i * nr_mics + mic].set(val))
            });
            Ok(res)
        }
        Err(_) => return Err(PyErr::new::<exc::BufferError, _>(py, "Execution failed!")),
    }
}
