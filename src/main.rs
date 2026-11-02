use rand::RngExt;
use ndarray::{Array6, Array4, Array2, Array1, ArrayD, Ix2, Ix4, Zip, Axis, s, Slice};
use ndarray_npy::NpzReader;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut npz = NpzReader::new(File::open("pokemon_dataset_split.npz")?)?;
    let x_train: Array4<f32> = npz.by_name("x_train.npy")?;
    let y_train: Array1<i32> = npz.by_name("y_train.npy")?;
    let x_val: Array4<f32> = npz.by_name("x_val.npy")?;
    let y_val: Array1<i32> = npz.by_name("y_val.npy")?;
    
    let total_train = x_train.shape()[0];
    let total_val = x_val.shape()[0];
    let num_classes = 150;

    println!("=========================================");
    println!("? Rust Dataset Load Completed!");
    println!("Train Images: {:?}", x_train.shape());
    println!("Val Images: {:?}", x_val.shape());
    println!("=========================================");

    let mut train_labels_onehot = Array2::<f32>::zeros((total_train, num_classes));
    for i in 0..total_train {
        let class_idx = y_train[i] as usize;
        train_labels_onehot[[i, class_idx]] = 1.0;
    }
    let mut val_labels_onehot = Array2::<f32>::zeros((total_val, num_classes));
    for i in 0..total_val {
        let class_idx = y_val[i] as usize;
        val_labels_onehot[[i, class_idx]] = 1.0;
    }

    let mut config = Config::default();
    config.optimizer = OptimizerKind::Adam;
    config.lr = 0.001;
    config.drop_ratio = 0.25;
    
    let mut net = Network::new(config);

    let input_flatten_size = 128 * 128 * 3;
    let hidden_layers_structure = vec![512, 256];
    
    net.set(input_flatten_size, hidden_layers_structure, num_classes);

    println!("? Layer auto generating completed! Deep learning start!");
    net.fit(
        x_train.into_dyn(),
        train_labels_onehot,
        10,
        32,
        true,
        1
    );
    let accuracy = net.accuracy(x_val.into_dyn(), val_labels_onehot);
    println!("\n=========================================");
    println!("? Train completed! final accuracy: {:.2}%", accuracy * 100.0);
    println!("=========================================");

    Ok(())
}
// ====================
// Support Function
// ====================
fn random_matrix(rows: usize, cols:usize) -> Vec<f32> {
    let mut rng = rand::rng();

    (0..(rows * cols))
        .map(|_| rng.random::<f32>())
        .collect()
}
fn he_matrix(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    let limit = (6.0 / rows as f32).sqrt();
    Array2::from_shape_fn((rows, cols), |_| (rng.random::<f32>() * 2.0 - 1.0) * limit)
}
fn sign(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}
fn im2col(x: Array4<f32>, fh: usize, fw: usize, stride: usize, pad: usize) -> Array2<f32> {
    let (n, c, h, w) = x.dim();
    let oh = (h + 2 * pad - fh) / stride + 1;
    let ow = (w + 2 * pad - fw) / stride + 1;
    let mut padded = Array4::<f32>::zeros((n, c, h + pad * 2, w + pad * 2));
    padded.slice_mut(s![.., .., pad..pad + h, pad..pad + w]).assign(&x);

    let mut col6 = Array6::<f32>::zeros((n, c, fh, fw, oh, ow));
    let st = stride as isize;
    for p in 0..fh {
        let p_max = p + stride * oh;
        for q in 0..fw {
            let q_max = q + stride * ow;
            let view = padded.slice(s![.., .., p..p_max;st, q..q_max;st]);
            col6.slice_mut(s![.., .., p, q, .., ..]).assign(&view);
        }
    }
    let permuted = col6.permuted_axes([0, 4, 5, 1, 2, 3]);
    permuted.as_standard_layout()
        .into_owned()
        .into_shape((n * oh * ow, c * fh  * fw))
        .unwrap()
}
fn col2im(col: Array2<f32>, x_shape: (usize, usize, usize, usize), fh: usize, fw: usize, stride: usize, pad: usize) -> Array4<f32> {
    let (n, c, h, w) = x_shape;
    let oh = (h + 2 * pad - fh) / stride + 1;
    let ow = (w + 2 * pad - fw) / stride + 1;
    let st = stride as isize;

    let col6 = col.view().into_shape((n, oh, ow, c, fh, fw)).unwrap().permuted_axes([0, 3, 4, 5, 1, 2]);

    let mut padded = Array4::<f32>::zeros((n, c, h + 2 * pad, w + 2 * pad));
    for p in 0..fh {
        let p_max = p + stride * oh;
        for q in 0..fw {
            let q_max = q + stride * ow;
            let mut target = padded.slice_mut(s![.., .., p..p_max;st, q..q_max;st]);
            target += &col6.slice(s![.., .., p, q, .., ..]);
        }
    }
    padded.slice(s![.., .., pad..pad + h, pad..pad + w]).to_owned()
}
// ====================
// Optimize Function
// ====================
trait Optimizer {
    fn update(&mut self, w: Array2<f32>, b: Array1<f32>, dw: Array2<f32>, db: Array1<f32>) -> (Array2<f32>, Array1<f32>);
}
#[derive(Clone)]
struct SGD {
    lr: f32,
    lambda1: f32,
    lambda2: f32
}
#[derive(Clone)]
struct Momentum {
    momentum: f32,
    lr: f32,
    vw: Option<Array2<f32>>,
    vb: Option<Array1<f32>>,
    lambda1: f32,
    lambda2: f32
}
#[derive(Clone)]
struct Adagrad {
    lr: f32,
    hw: Option<Array2<f32>>,
    hb: Option<Array1<f32>>,
    lambda1: f32,
    lambda2: f32
}
#[derive(Clone)]
struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    mw: Option<Array2<f32>>,
    mb: Option<Array1<f32>>,
    vw: Option<Array2<f32>>,
    vb: Option<Array1<f32>>,
    t: i32,
    lambda1: f32,
    lambda2: f32
}

impl SGD {
    fn new(lr: f32, lambda1: f32, lambda2: f32) -> Self {
        Self { lr, lambda1, lambda2 }
    }
}
impl Optimizer for SGD {
    fn update(&mut self, mut w: Array2<f32>, mut b: Array1<f32>, mut dw: Array2<f32>, db: Array1<f32>) -> (Array2<f32>, Array1<f32>) {
        let w_sign = w.mapv(sign);
        dw = dw + (self.lambda1 * &w_sign) + (self.lambda2 * &w);
        w = w - (self.lr * &dw);
        b = b - (self.lr * &db);

        (w, b)
    }
}
impl Momentum {
    fn new(lr: f32, lambda1: f32, lambda2: f32, momentum: f32) -> Self {
        Self { lr, vw: None, vb: None, lambda1, lambda2, momentum }
    }
}
impl Optimizer for Momentum {
    fn update(&mut self, mut w: Array2<f32>, mut b: Array1<f32>, mut dw: Array2<f32>, db: Array1<f32>) -> (Array2<f32>, Array1<f32>) {
        if self.vw.is_none() {
            self.vw = Some(Array2::zeros(w.dim()))
        }
        if self.vb.is_none() {
            self.vb = Some(Array1::zeros(b.dim()))
        }

        let w_sign = w.mapv(sign);
        dw = dw + (self.lambda1 * &w_sign) + (self.lambda2 * &w);

        if let (Some(vw), Some(vb)) = (&mut self.vw, &mut self.vb) {
            *vw = (self.momentum * &*vw) - (self.lr * &dw);
            w = w + &*vw;
            *vb = (self.momentum * &*vb) - (self.lr * &db);
            b = b + &*vb;
        }

        (w, b)
    }
}
impl Adagrad {
    fn new(lr: f32, lambda1: f32, lambda2: f32) -> Self {
        Self { lr, hw: None, hb: None, lambda1, lambda2 }
    }
}
impl Optimizer for Adagrad {
    fn update(&mut self, mut w: Array2<f32>, mut b: Array1<f32>, mut dw: Array2<f32>, db: Array1<f32>) -> (Array2<f32>, Array1<f32>) {
        if self.hw.is_none() {
            self.hw = Some(Array2::zeros(w.dim()))
        }
        if self.hb.is_none() {
            self.hb = Some(Array1::zeros(b.dim()))
        }

        let w_sign = w.mapv(sign);
        dw = dw + (self.lambda1 * &w_sign) + (self.lambda2 * &w);

        if let (Some(hw), Some(hb)) = (&mut self.hw, &mut self.hb) {
            Zip::from(hw)
                .and(&mut w)
                .and(&dw)
                .for_each(|h_val, w_val, &dw_val| {
                    *h_val += dw_val * dw_val;
                    *w_val -= dw_val * self.lr / (h_val.sqrt() + 1e-7);
                });
            Zip::from(hb)
                .and(&mut b)
                .and(&db)
                .for_each(|h_val, b_val, &db_val| {
                    *h_val += db_val * db_val;
                    *b_val -= db_val * self.lr / (h_val.sqrt() + 1e-7);
                });
        }

        (w, b)
    }
}
impl Adam {
    fn new(lr: f32, lambda1: f32, lambda2: f32, beta1: f32, beta2: f32) -> Self {
        Self { lr, mw: None, mb: None, vw: None, vb: None, t: 0, lambda1, lambda2, beta1, beta2 }
    }
}
impl Optimizer for Adam {
    fn update(&mut self, mut w: Array2<f32>, mut b: Array1<f32>, mut dw: Array2<f32>, db: Array1<f32>) -> (Array2<f32>, Array1<f32>) {
        if self.mw.is_none() {
            self.mw = Some(Array2::zeros(w.dim()))
        }
        if self.mb.is_none() {
            self.mb = Some(Array1::zeros(b.dim()))
        }
        if self.vw.is_none() {
            self.vw = Some(Array2::zeros(w.dim()))
        }
        if self.vb.is_none() {
            self.vb = Some(Array1::zeros(b.dim()))
        }

        self.t += 1;
        let w_sign = w.mapv(sign);
        dw = dw + (self.lambda1 * &w_sign) + (self.lambda2 * &w);

        if let (Some(mw), Some(mb), Some(vw), Some(vb)) = (&mut self.mw, &mut self.mb, &mut self.vw, &mut self.vb) {
            Zip::from(mw)
                .and(vw)
                .and(&mut w)
                .and(&dw)
                .for_each(|m_val, v_val, w_val, &dw_val| {
                    *m_val = (self.beta1 * *m_val) + (1.0 - self.beta1) * dw_val;
                    *v_val = (self.beta2 * *v_val) + (1.0 - self.beta2) * (dw_val * dw_val);

                    let m_hat = *m_val / (1.0 - self.beta1.powi(self.t));
                    let v_hat = *v_val / (1.0 - self.beta2.powi(self.t));
                    *w_val -= (self.lr * m_hat) / (v_hat.sqrt() + 1e-7);
                });

            Zip::from(mb)
                .and(vb)
                .and(&mut b)
                .and(&db)
                .for_each(|m_val, v_val, b_val, &db_val| {
                    *m_val = (self.beta1 * *m_val) + (1.0 - self.beta1) * db_val;
                    *v_val = (self.beta2 * *v_val) + (1.0 - self.beta2) * (db_val * db_val);

                    let m_hat = *m_val / (1.0 - self.beta1.powi(self.t));
                    let v_hat = *v_val / (1.0 - self.beta2.powi(self.t));
                    *b_val -= (self.lr * m_hat) / (v_hat.sqrt() + 1e-7);
                });
        }

        (w, b)
    }
}
// ====================
// Active Function
// ====================
struct ReLU {
    mask: Option<ArrayD<bool>>
}
struct Sigmoid {
    y: Option<Array2<f32>>
}
struct Softmax {
    y: Option<Array2<f32>>
}

impl ReLU {
    fn new() -> Self {
        Self { mask: None }
    }
    fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        self.mask = Some(x.mapv(|e| e <= 0.0));
        x.mapv(|e| e.max(0.0))
    }
    fn backward(&mut self, dout: ArrayD<f32>) -> ArrayD<f32> {
        let mask = self.mask.as_ref().expect("Forward must be called before backward");
        Zip::from(&dout).and(mask).map_collect(|&d, &m| if m { 0.0 } else { d })
    }
}
impl Sigmoid {
    fn new() -> Self {
        Self { y: None }
    }
    fn forward(&mut self, x_list: Array2<f32>) -> Array2<f32> {
        let out = x_list.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.y = Some(out.clone());
        out
    }
    fn backward(&mut self, dout: Array2<f32>) -> Array2<f32> {
        let y = self.y.as_ref().expect("Forward must be called before backward");
        Zip::indexed(dout.view()).map_collect(|(i, j), &e| {
            e * y[[i, j]] * (1.0 - y[[i, j]])
        })
    }
}
impl Softmax {
    fn new() -> Self {
        Self { y: None }
    }
    fn forward(&mut self, mut x: Array2<f32>) -> Array2<f32> {
        let max_per_row = x.map_axis(Axis(1), |v| {
            v.fold(f32::NEG_INFINITY, |max_val, &e| max_val.max(e))
        });

        for (mut row, &max_val) in x.axis_iter_mut(Axis(0)).zip(max_per_row.iter()) {
            row -= max_val;
        }

        x.mapv_inplace(|v| v.exp());
        let sum_per_row = x.sum_axis(Axis(1));

        for (mut row, &sum_val) in x.axis_iter_mut(Axis(0)).zip(sum_per_row.iter()) {
            row /= sum_val;
        }

        self.y = Some(x.clone());
        x
    }
    fn backward(&mut self, dout: Array2<f32>) -> Array2<f32> {
        let y = self.y.as_ref().expect("Forward must be called before backward");

        Zip::indexed(dout.view()).map_collect(|(i, j), _| {
            let dout_row = dout.row(i);
            dout_row.indexed_iter().fold(0.0, |acc, (k, &e)| {
                let gradient = if j == k {
                    y[[i, j]] * (1.0 - y[[i, j]])
                } else {
                    - y[[i, j]] * y[[i, k]]
                };

                acc + e * gradient
            })
        })
    }
}
// ====================
// Fully Connected Layer
// ====================
struct Affine {
    x: Option<Array2<f32>>,
    w: Array2<f32>,
    b: Array1<f32>,
    dw: Option<Array2<f32>>,
    db: Option<Array1<f32>>,
    optimizer: Box<dyn Optimizer>
}
struct BatchNormalization {
    x_hat: Option<Array2<f32>>,
    sigma: Option<Array1<f32>>,
    gamma: Array1<f32>,
    beta: Array1<f32>,
    dgamma: Option<Array1<f32>>,
    dbeta: Option<Array1<f32>>,
    lr: f32
}

impl Affine {
    fn new(w: Array2<f32>, b: Array1<f32>, optimizer: Box<dyn Optimizer>) -> Self {
        Self { x: None, dw: None, db: None, w, b, optimizer }
    }
    fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        let out = x.dot(&self.w) + &self.b;
        self.x = Some(x);
        out
    }
    fn backward(&mut self, dout: Array2<f32>) -> Array2<f32> {
        let x = self.x.as_ref().expect("Forward must be called before backward");

        self.dw = Some(x.t().dot(&dout));
        self.db = Some(dout.sum_axis(Axis(0)));

        let dx = dout.dot(&self.w.t());

        dx
    }
    fn step(&mut self) {
        let w = std::mem::take(&mut self.w);
        let b = std::mem::take(&mut self.b);
        let dw = self.dw.take().expect("backward first");
        let db = self.db.take().expect("backward first");
        let (w, b) = self.optimizer.update(w, b, dw, db);
        self.w = w;
        self.b = b;
    }
}
impl BatchNormalization {
    fn new(size: usize, lr: f32) -> Self {
        Self { x_hat: None, sigma: None, gamma: Array1::<f32>::ones(size), beta: Array1::<f32>::zeros(size), dgamma: None, dbeta: None, lr }
    }
    fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
        let mu = x.sum_axis(Axis(0)) / (x.nrows() as f32);
        let sigma = (&x - &mu).mapv(|v| v.powi(2)).sum_axis(Axis(0)) / (x.nrows() as f32);
        let x_hat = (&x - &mu) / (&sigma + 1e-7).mapv(|v| v.sqrt());
        
        self.x_hat = Some(x_hat.clone());
        self.sigma = Some(sigma.clone());
        
        &x_hat * &self.gamma + &self.beta
    }
    fn backward(&mut self, dout: Array2<f32>) -> Array2<f32> {
        let x_hat = self.x_hat.as_ref().expect("Forward must be called before backward");
        let sigma = self.sigma.as_ref().expect("Forward must be called before backward");
        
        let m = dout.nrows();
        let n = dout.ncols();
        
        let mut dx = Array2::<f32>::zeros((m, n));
        let mut dgamma = Array1::<f32>::zeros(n);
        let mut dbeta = Array1::<f32>::zeros(n);
        
        for j in 0..n {
            let sum_d = dout.column(j).sum();
            let sum_dx = Zip::from(dout.column(j)).and(x_hat.column(j)).fold(0.0, |acc, &d, &x| acc + d * x);
            for i in 0..m {
                dx[[i, j]] = self.gamma[j] / (sigma[j] + 1e-7).sqrt() * (dout[[i, j]] - sum_d / m as f32 - x_hat[[i, j]] / m as f32 * sum_dx);
            }
            dgamma[j] = Zip::from(dout.column(j)).and(x_hat.column(j)).fold(0.0, |acc, &a, &b| acc + a * b);
            dbeta[j] = dout.t().row(j).iter().sum();
        }
        self.dgamma = Some(dgamma);
        self.dbeta = Some(dbeta);
        dx
    }
    fn step(&mut self) {
        let dgamma = self.dgamma.take().expect("backward first");
        let dbeta = self.dbeta.take().expect("backward first");
        
        self.gamma = &self.gamma - self.lr * &dgamma;
        self.beta = &self.beta - self.lr * &dbeta;
    }
}
// ====================
// Loss Function
// ====================
struct CrossEntropy {
    p: Option<Array2<f32>>,
    q: Option<Array2<f32>>
}

impl CrossEntropy {
    fn new() -> Self {
        Self { p: None, q: None }
    }
    fn forward(&mut self, p: Array2<f32>, q: Array2<f32>) -> Array1<f32> {
        self.p = Some(p.clone());
        self.q = Some(q.clone());
        -(&q * &p.mapv(|v| (v + 1e-7).ln())).sum_axis(Axis(1))
    }
    fn backward(&mut self, dout: Array1<f32>) -> Array2<f32> {
        let p = self.p.as_ref().expect("Forward must be called before backward");
        let q = self.q.as_ref().expect("Forward must be called before backward");
        -(&dout.insert_axis(Axis(1)) * q / &(p + 1e-7))
    }
}
// ====================
// Output Layer
// ====================
struct SoftmaxWithLoss {
    softmax: Softmax,
    cee: CrossEntropy,
    p: Option<Array2<f32>>,
    q: Option<Array2<f32>>
}
struct DropOut {
    drop_ratio: f32,
    scale: f32,
    mask: Option<Array2<bool>>
}

impl SoftmaxWithLoss {
    fn new(softmax: Softmax, cee: CrossEntropy) ->  Self {
        Self { p: None, q: None, softmax, cee }
    }
    fn forward(&mut self, x: Array2<f32>, q: Array2<f32>) -> Array1<f32> {
        let p = self.softmax.forward(x);
        let loss = self.cee.forward(p.clone(), q.clone());

        self.p = Some(p);
        self.q = Some(q);

        loss
    }
    fn backward(&self) -> Array2<f32> {
        let p = self.p.as_ref().expect("Forward must be called before backward");
        let q = self.q.as_ref().expect("Forward must be called before backward");

        (p - q) / p.nrows() as f32
    }
}
impl DropOut {
    fn new(drop_ratio: f32) -> Self {
        assert!((0.0..1.0).contains(&drop_ratio));
        Self { drop_ratio, scale: 1.0 / (1.0 - drop_ratio), mask: None }
    }
    fn forward(&mut self, x: Array2<f32>, train: bool) -> Array2<f32> {
        if !train {
            x
        } else {
            let mut rng = rand::rng();
            let mask = x.mapv(|_| rng.random::<f32>() > self.drop_ratio);
            self.mask = Some(mask.clone());
            x * mask.mapv(|v| v as u8 as f32) * self.scale
        }
    }
    fn backward(&mut self, dout: Array2<f32>) -> Array2<f32> {
        let mask = self.mask.as_ref().expect("Forward must be called before backward");
        dout * mask.mapv(|v| v as u8 as f32) * self.scale
    }
}
// ====================
// Network
// ====================
enum Layer {
    Affine(Affine),
    ReLU(ReLU),
    DropOut(DropOut),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
    CrossEntropy(CrossEntropy),
    SoftmaxWithLoss(SoftmaxWithLoss),
    BatchNormalization(BatchNormalization),
    Convolution(Convolution),
    MaxPooling(MaxPooling),
    Flatten(Flatten)
}

enum OptimizerKind { SGD, Momentum, Adagrad, Adam }

impl Layer {
    fn forward(&mut self, x: ArrayD<f32>, train: bool) -> ArrayD<f32> {
        match self {
            Layer::Affine(l) => l.forward(x.into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            Layer::DropOut(l) => l.forward(x.into_dimensionality::<Ix2>().unwrap(), train).into_dyn(),
            Layer::Sigmoid(l) => l.forward(x.into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            Layer::BatchNormalization(l) => l.forward(x.into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            Layer::Convolution(l) => l.forward(x.into_dimensionality::<Ix4>().unwrap()).into_dyn(),
            Layer::MaxPooling(l) => l.forward(x.into_dimensionality::<Ix4>().unwrap()).into_dyn(),
            Layer::Flatten(l) => l.forward(x.into_dimensionality::<Ix4>().unwrap()).into_dyn(),
            Layer::ReLU(l) => l.forward(x),
            _ => x
        }
    }
    fn backward(&mut self, dout: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            Layer::Affine(l) => l.backward(dout.into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            Layer::DropOut(l) => l.backward(dout.into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            Layer::Sigmoid(l) => l.backward(dout.into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            Layer::BatchNormalization(l) => l.backward(dout.into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            Layer::Convolution(l) => l.backward(dout.into_dimensionality::<Ix4>().unwrap()).into_dyn(),
            Layer::MaxPooling(l) => l.backward(dout.into_dimensionality::<Ix4>().unwrap()).into_dyn(),
            Layer::Flatten(l) => l.backward(dout.into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            Layer::ReLU(l) => l.backward(dout),
            _ => dout
        }
    }
}
struct Config {
    layers: Vec<(String, Layer)>,
    last_layer: Option<SoftmaxWithLoss>,
    optimizer: OptimizerKind,
    lr: f32,
    lambda1: f32,
    lambda2: f32,
    beta1: f32,
    beta2: f32,
    momentum: f32,
    drop_ratio: f32,
}
struct Network {
    config: Config
}
impl Default for Config {
    fn default() -> Self {
        Self { layers: Vec::new(), last_layer: None, optimizer: OptimizerKind::SGD, lr: 0.01, lambda1: 0.001, lambda2: 0.001, beta1: 0.9, beta2: 0.999, momentum: 0.9, drop_ratio: 0.0 }
    }
}
impl Config {
    fn make_optimizer(&self) -> Box<dyn Optimizer> {
        match self.optimizer {
            OptimizerKind::SGD => Box::new(SGD::new(self.lr, self.lambda1, self.lambda2)),
            OptimizerKind::Momentum => Box::new(Momentum::new(self.lr, self.lambda1, self.lambda2, self.momentum)),
            OptimizerKind::Adagrad => Box::new(Adagrad::new(self.lr, self.lambda1, self.lambda2)),
            OptimizerKind::Adam => Box::new(Adam::new(self.lr, self.lambda1, self.lambda2, self.beta1, self.beta2)),
        }
    }
}
impl Network {
    fn new(config: Config) -> Self {
        Self { config }
    }
    fn set(&mut self, input_size: usize, hidden_size: Vec<usize>, output_size: usize) {
        let mut layers: Vec<(String, Layer)> = Vec::new();

        layers.push(("Flatten".to_string(), Layer::Flatten(Flatten::new())));
        
        let mut sizes = vec![input_size];
        sizes.extend(hidden_size);
        sizes.push(output_size);
        for (n, window) in sizes.windows(2).enumerate() {
            let cols = window[1];

            let matrix = he_matrix(window[0], window[1]);
            let bias = Array1::zeros(cols);

            let affine_struct = Affine::new(matrix, bias, self.config.make_optimizer());
            let affine_layer = Layer::Affine(affine_struct);

            layers.push((format!("Affine{}", n + 1), affine_layer));
            if n != sizes.len() - 2 {
                let bn_struct = BatchNormalization::new(cols, self.config.lr);
                let bn_layer = Layer::BatchNormalization(bn_struct);
                
                let relu_struct = ReLU::new();
                let relu_layer = Layer::ReLU(relu_struct);

                let dropout_struct = DropOut::new(self.config.drop_ratio);
                let dropout_layer = Layer::DropOut(dropout_struct);
                
                layers.push((format!("BatchNormalization{}", n + 1), bn_layer));
                layers.push((format!("ReLU{}", n + 1), relu_layer));
                layers.push((format!("DropOut{}", n + 1), dropout_layer));
            }
        }
        let softmax = Softmax::new();
        let cee = CrossEntropy::new();
        let softmaxwithloss_layer = SoftmaxWithLoss::new(softmax, cee);
        self.config.layers = layers;
        self.config.last_layer = Some(softmaxwithloss_layer);
    }
    fn forward(&mut self, mut batch_x: ArrayD<f32>, batch_y: Array2<f32>, train: bool) -> Array1<f32> {
        for (_, layer) in self.config.layers.iter_mut() {
            batch_x = layer.forward(batch_x, train);
        }
        let loss_layer = self.config.last_layer.as_mut().unwrap();
        loss_layer.forward(batch_x.into_dimensionality::<Ix2>().unwrap(), batch_y)
    }
    fn backward(&mut self) {
        let loss_layer = self.config.last_layer.as_mut().unwrap();
        let mut dout = loss_layer.backward().into_dyn();
        for (_, layer) in self.config.layers.iter_mut().rev() {
            dout = layer.backward(dout);
        }
    }
    fn fit(&mut self, x: ArrayD<f32>, y: Array2<f32>, epochs: usize, batch_size: usize, verbose: bool, log_interval: usize) {
        let mut total_loss = 0.0_f32;
        for epoch in 0..epochs {
            for i in (0..x.shape()[0]).step_by(batch_size) {
                let end = (i + batch_size).min(x.shape()[0]);
                let batch_x = x.slice_axis(Axis(0), Slice::from(i..end)).to_owned();
                let batch_y = y.slice(s![i..end, ..]).to_owned();
                let loss = self.forward(batch_x, batch_y, true);

                let mut reg = 0.0_f32;
                if self.config.lambda1 != 0.0 || self.config.lambda2 != 0.0 {
                    let mut l1 = 0.0;
                    let mut l2 = 0.0;

                    for (_, layer) in self.config.layers.iter() {
                        if let Layer::Affine(a) = layer {
                            l1 += a.w.iter().map(|v| v.abs()).sum::<f32>();
                            l2 += a.w.iter().map(|v| v * v).sum::<f32>();
                        }
                    }

                    reg = self.config.lambda1 * l1 + self.config.lambda2 / 2.0 * l2;
                }
                total_loss = loss.mean().unwrap() + reg;
                self.backward();
                for (_, layer) in self.config.layers.iter_mut() {
                    match layer {
                        Layer::Affine(a) => a.step(),
                        Layer::BatchNormalization(bn) => bn.step(),
                        Layer::Convolution(c) => c.step(),
                        _ => {}
                    }
                }
            }
            if verbose && epoch % log_interval == 0 {
                println!("epoch {}: {}", epoch, total_loss);
            }
        }
    }
    fn predict(&mut self, mut x: ArrayD<f32>) -> Array2<f32> {
        for (_, layer) in self.config.layers.iter_mut() {
            x = layer.forward(x, false);
        }
        self.config.last_layer.as_mut().unwrap().softmax.forward(x.into_dimensionality::<Ix2>().unwrap())
    }
    fn accuracy(&mut self, x: ArrayD<f32>, t: Array2<f32>) -> f32 {
        let y = self.predict(x.clone());
        let mut correct = 0.0;
        
        for i in 0..x.shape()[0] {
            let predicted = y.row(i).iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            let answer = t.row(i).iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            
            if predicted == answer {
                correct += 1.0;
            }
        }
        correct / x.shape()[0] as f32
    }
}

struct Convolution {
    col: Option<Array2<f32>>,
    x_shape: Option<(usize, usize, usize, usize)>,
    w: Array4<f32>,
    b: Array1<f32>,
    stride: usize,
    pad: usize,
    dw: Option<Array4<f32>>,
    db: Option<Array1<f32>>,
    optimizer: Box<dyn Optimizer>
}

impl Convolution {
    fn new(w: Array4<f32>, b: Array1<f32>, stride: usize, pad: usize, optimizer: Box<dyn Optimizer>) ->  Self {
        Self {
            col: None,
            x_shape: None,
            w,
            b,
            stride,
            pad,
            dw: None,
            db: None,
            optimizer
        }
    }
    fn forward(&mut self, x: Array4<f32>) -> Array4<f32> {
        assert!(self.stride > 0);
        let (fn_, c, fh, fw) = self.w.dim();
        self.x_shape = Some(x.dim());
        let (n, c_, h, w) = self.x_shape.unwrap();
        assert!(c == c_);
        let out_h = 1 + (h + 2 * self.pad - fh) / self.stride;
        let out_w = 1 + (w + 2 * self.pad - fw) / self.stride;
        let col = im2col(x, fh, fw, self.stride, self.pad);
        let row_w = self.w.view().into_shape((fn_, c * fw * fh)).unwrap();
        let col_w = row_w.t();
        let out = col.dot(&col_w) + &self.b;
        let mut new_out = out.into_shape((n, out_h, out_w, fn_)).unwrap();
        new_out = new_out.permuted_axes([0, 3, 1, 2]).as_standard_layout().into_owned();
        self.col = Some(col);
        new_out
    }
    fn backward(&mut self, dout: Array4<f32>) -> Array4<f32> {
        let col = self.col.as_ref().expect("Forward must be called before backward");
        let (fn_, c, fh, fw) = self.w.dim();
        let (n, dfn, oh, ow) = dout.dim();
        assert!(fn_ == dfn);
        let dy = dout.permuted_axes([0, 2, 3, 1]).as_standard_layout().into_owned().into_shape((n * oh * ow, fn_)).unwrap();
        let db = dy.sum_axis(Axis(0));
        let dw_col: Array2<f32> = col.t().dot(&dy);
        let dw = dw_col.t().as_standard_layout().into_owned().into_shape((fn_, c, fh, fw)).unwrap();

        let row_w = self.w.view().into_shape((fn_, c * fw * fh)).unwrap();
        let dcol = dy.dot(&row_w);
        self.dw = Some(dw);
        self.db = Some(db);
        col2im(dcol, self.x_shape.unwrap(), fh, fw, self.stride, self.pad)
    }
    fn step(&mut self) {
        let (fn_, c, fh, fw) = self.w.dim();
        let dw = self.dw.take().expect("backward first");
        let db = self.db.take().expect("backward first");

        let w2 = std::mem::take(&mut self.w).into_shape((fn_, c * fh * fw)).unwrap();
        let dw2 = dw.into_shape((fn_, c * fh * fw)).unwrap();
        let b = std::mem::take(&mut self.b);
        let (w2, b) = self.optimizer.update(w2, b, dw2, db);
        self.w = w2.into_shape((fn_, c, fh, fw)).unwrap();
        self.b = b;
    }
}

struct MaxPooling {
    arg_max: Option<Array4<(usize, usize)>>,
    x_shape: Option<(usize, usize, usize, usize)>,
    pool_h: usize,
    pool_w: usize,
    stride: usize
}

impl MaxPooling {
    fn new(pool_h: usize, pool_w: usize, stride: usize) -> Self {
        Self {
            arg_max: None,
            x_shape: None,
            pool_h,
            pool_w,
            stride
        }
    }
    fn forward(&mut self, x: Array4<f32>) -> Array4<f32> {
        assert!(self.stride > 0);
        assert!(self.pool_h > 0 && self.pool_w > 0);
        
        self.x_shape = Some(x.dim());
        let (n, c, h, w) = self.x_shape.unwrap();

        assert!(self.pool_h <= h && self.pool_w <= w);
        
        let oh = (h - self.pool_h) / self.stride + 1;
        let ow = (w - self.pool_w) / self.stride + 1;

        let mut arg_max = Array4::<(usize, usize)>::default((n, c, oh, ow));
        let mut out = Array4::<f32>::zeros((n, c, oh, ow));
        for b in 0..n {
            for ch in 0..c {
                for i in 0..oh {
                    let h_start = i * self.stride;
                    for j in 0..ow {
                        let w_start = j * self.stride;
                        let mut max = f32::NEG_INFINITY;
                        for ph in 0..self.pool_h {
                            for pw in 0..self.pool_w {
                                let vh = h_start + ph;
                                let vw = w_start + pw;
                                if x[[b, ch, vh, vw]] > max {
                                    max = x[[b, ch, vh, vw]];
                                    arg_max[[b, ch, i, j]] = (vh, vw);
                                    out[[b, ch, i, j]] = max;
                                }
                            }
                        }
                    }
                }
            }
        }
        self.arg_max = Some(arg_max);
        out
    }
    fn backward(&mut self, dout: Array4<f32>) -> Array4<f32> {
        let (n, c, h, w) = self.x_shape.unwrap();
        let oh = (h - self.pool_h) / self.stride + 1;
        let ow = (w - self.pool_w) / self.stride + 1;
        let mut dx = Array4::<f32>::zeros((n, c, h, w));
        let dout_idxs = self.arg_max.as_ref().unwrap();
        for b in 0..n {
            for ch in 0..c {
                for i in 0..oh {
                    for j in 0..ow {
                        let (vh, vw) = dout_idxs[[b, ch, i, j]];
                        dx[[b, ch, vh, vw]] += dout[[b, ch, i, j]];
                    }
                }
            }
        }
        dx
    }
}


struct Flatten {
    input_shape: Option<(usize, usize, usize, usize)>
}

impl Flatten {
    fn new() -> Self {
        Self { input_shape: None }
    }
    fn forward(&mut self, x: Array4<f32>) -> Array2<f32> {
        let (n, c, h, w) = x.dim();
        self.input_shape = Some((n, c, h, w));

        x.into_shape((n, c * h * w)).unwrap()
    }
    fn backward(&mut self, dout: Array2<f32>) -> Array4<f32> {
        let shape = self.input_shape.expect("Forward must be called before backward");

        dout.into_shape(shape).unwrap()
    }
}