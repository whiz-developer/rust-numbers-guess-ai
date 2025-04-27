use std::fs::File;
use std::io::{BufReader, BufWriter};
use image::{Luma};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Write};
use rand::prelude::SliceRandom;

fn get_filenames_in_dir(path: &str) -> Vec<String> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();

                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        files.push(name.to_string());
                    }
                }
            }
        }
    }

    files
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron{
    weights: Vec<f32>,
    bias: f32,
    output: f32,
    delta: f32
}

#[derive(Clone, Serialize, Deserialize)]
struct Layer{
    neurons:Vec<Neuron>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Network {
    layers: Vec<Layer>
}

impl Network {
    fn new(sizes: &[usize]) -> Network {
        let mut rng = rand::rng();
        let mut layers = Vec::new();

        for i in 1..sizes.len() {
            let prev_size = sizes[i - 1];
            let layer_size = sizes[i];
            let mut neurons = Vec::new();

            for _ in 0..layer_size {
                let std_dev = (2.0 / prev_size as f32).sqrt();
                let weights: Vec<f32> = (0..prev_size)
                    .map(|_| rng.random_range(-std_dev..std_dev))
                    .collect();

                let bias = 0.0;
                let neuron = Neuron { weights, bias, output: 0.0, delta: 0.0 };
                neurons.push(neuron);
            }

            layers.push(Layer { neurons });
        }

        Network { layers }
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut activations = input;
        for i in 0..self.layers.len() {
            let mut new_activations = Vec::new();
            for neuron in &mut self.layers[i].neurons {
                let sum: f32 = neuron.weights
                    .iter()
                    .zip(&activations)
                    .map(|(w, a)| w * a)
                    .sum::<f32>() + neuron.bias;

                neuron.output = relu(sum);
                new_activations.push(neuron.output);
            }
            activations = new_activations;
        }
        activations
    }

    fn train(&mut self, inputs: Vec<f32>, targets: Vec<f32>, lr: f32) -> f32 {
        let mut result_error = 0.0;

        let last = self.layers.len() - 1;
        for (i, neuron) in self.layers[last].neurons.iter_mut().enumerate() {
            let error = targets[i] - neuron.output;
            result_error += error * error;
            neuron.delta = error * relu_derivative(neuron.output);
        }

        result_error /= targets.len() as f32;

        for l in (0..last).rev() {
            for i in 0..self.layers[l].neurons.len() {
                let mut error = 0.0;

                for neuron_next in &self.layers[l + 1].neurons {
                    error += neuron_next.weights[i] * neuron_next.delta;
                }

                let neuron = &mut self.layers[l].neurons[i];
                neuron.delta = error * relu_derivative(neuron.output);
            }
        }

        let mut activations = inputs;

        for l in 0..self.layers.len() {
            let prev_activations = activations.clone();

            for neuron in &mut self.layers[l].neurons {
                for j in 0..neuron.weights.len() {
                    neuron.weights[j] += lr * neuron.delta * prev_activations[j];
                }
                neuron.bias += lr * neuron.delta;
            }

            activations = self.layers[l].neurons.iter().map(|n| n.output).collect();
        }

        result_error
    }


    fn save(&self, path: &str) {
        let file = File::create(path).unwrap();
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self).unwrap();
    }

    fn load(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).unwrap()
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}



fn main() {

    let mut network = Network::load("network.json");

    let mut rng = rand::rng();

    let mut files = get_filenames_in_dir("D:/Desktop/mnist/train");

    let num_epochs = 10;

    for epoch in 0..num_epochs {
        files.shuffle(&mut rng);
        println!("Epoch {}/{}", epoch, num_epochs);

        for batch_start in (0..files.len()).step_by(100) {
            let batch_end = (batch_start + 100).min(files.len());
            let batch = &files[batch_start..batch_end];

            for file in batch {
                let parsed_image = parse_image(format!("D:/Desktop/mnist/train/{file}"));
                network.train(parsed_image, one_hot(extract_label(file), 10), 0.0001);
            }

            print!("\r{}", (batch_end as f32 / files.len() as f32) * 100.0);
            std::io::stdout().flush().unwrap();

            network.save("network.json");
        }
        println!();

        println!("Testing model...");
        let mut test_files = get_filenames_in_dir("D:/Desktop/mnist/test");
        let mut right_ans = 0;
        let mut wrong_ans = 0;

        for file in &test_files {
            print!("\r{}", ((right_ans + wrong_ans) as f32 / test_files.len() as f32) * 100.0);
            std::io::stdout().flush().unwrap();


            let parsed_image = parse_image(format!("D:/Desktop/mnist/test/{file}"));
            let ans = extract_label(&file);
            let guessed = network.forward(parsed_image).iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _value)| index)
                .unwrap();
            if ans == guessed {
                right_ans+=1;
            } else {
                wrong_ans+=1;
            }
        }
        println!();
        println!("Model tested!");
        println!("{}/{}", right_ans, right_ans+wrong_ans);

    }
}



fn forward_test() {
    let mut network = Network::load("network.json");
    let parsed_image = parse_image("input.png".to_string());
    let res = network.forward(parsed_image);

    visualize_output(res);
}



fn visualize_output(output: Vec<f32>) {
    for (i, &value) in output.iter().enumerate() {
        let num_hashes = (value * 50.0) as usize;
        let bar = "#".repeat(num_hashes);
        println!("Number {}: {:>6.2} | {}", i, value, bar);
    }
}


fn one_hot(label: usize, size: usize) -> Vec<f32> {
    let mut vec = vec![0.0; size];
    vec[label] = 1.0;
    vec
}


fn extract_label(filename: &str) -> usize {
    filename.split('_')
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .expect("Невозможно распарсить метку")
}

fn parse_image(path: String) -> Vec<f32>{
    let mut image_vec: Vec<f32> = Vec::new();

    let img = image::open(path)
        .expect("Can't open the file")
        .to_luma8();

    let (width, height) = img.dimensions();

    for y in 0..height {
        for x in 0..width {
            let pixel: &Luma<u8> = img.get_pixel(x, y);
            let brightness = pixel[0] as f32 / 255.0;
            image_vec.push(brightness);
        }
    }
    image_vec
}
