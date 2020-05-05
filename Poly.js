let x_vals = [];
let y_vals = [];

let a, b, c,d,e;
let dragging = false;

const learningRate = 0.09;
const optimizer = tf.train.adam(learningRate);

function setup() {
    var cnv = createCanvas(700, 300);
    var x = (windowWidth - width) / 2;
    var y = (windowHeight - height) / 2;
    cnv.position(x, y);
    background(255, 0, 200);
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
  e = tf.variable(tf.scalar(random(-1, 1)));
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // the equation--===== ax^4 + bx^3 + cx^2 + dx+e
  const ys = xs
    .pow(tf.scalar(4))
    .mul(a).add(xs.pow(tf.scalar(3)).mul(b))
    .add(xs.square().mul(c))
    .add(xs.mul(d))
    .add(e);
  return ys;
}

function mousePressed() {
  dragging = true;
}

function mouseReleased() {
  dragging = false;
}

function draw() {
  if (dragging) {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
  } else {
    tf.tidy(() => {
      if (x_vals.length > 0) {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(x_vals), ys));
      }
    });
  }

  background(0);

  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }

  const curveX = [];
  for (let x = -1; x <= 1; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }
  endShape();

  // console.log(tf.memory().numTensors);
}
