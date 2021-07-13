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
                seed: u64
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
    seed: u64,
) -> PyResult<usize> {
    let h_buf = match buffer::PyBuffer::get(py, &h_py) {
        Ok(buffer) => buffer,
        Err(err) => return Err(err),
    };
    let mut h: Vec<Vec<f64>> = Vec::new();
    for _ in 0..h_buf.shape()[1] {
        h.push(vec![0.0; nt]);
    }

    let xs = match buffer::PyBuffer::get(py, &xs_py) {
        Ok(buffer) => buffer.to_vec::<f64>(py).unwrap(),
        Err(err) => return Err(err),
    };

    let xr = match buffer::PyBuffer::get(py, &xr_py) {
        Ok(buffer) => buffer.to_vec::<f64>(py).unwrap(),
        Err(err) => return Err(err),
    };

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

    match rim::solve_rim(&mut h, xs, xr, l, beta, n, rd, nt, seed) {
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
