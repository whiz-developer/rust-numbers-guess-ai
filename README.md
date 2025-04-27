# Rust Number Guessing AI

This is a small, basic AI example written in Rust that attempts to guess a number from a 28x28 pixel image. It's my first foray into the world of Machine Learning! This project is intended as a learning experience and a starting point for more complex AI explorations.

**Note:** The repository includes a pre-trained model (`network.json`) for immediate testing. You can also train your own model as described below.

## How to use?

### 1. Prerequisites

*   **Rust Toolchain:** Make sure you have Rust installed. You can download it from [https://www.rust-lang.org/](https://www.rust-lang.org/).

### 2. Training the AI (Optional)

By default, the `main` function is set up to train the AI. This step is optional as a pre-trained model is already included. To train the AI:

1.  **Dataset:** The code expects a directory structure with training and testing images. You'll need to obtain a MNIST dataset (or a similar dataset of handwritten digits) and organize the images accordingly in `train` and `test` directories. Rename image files to be in the format `{label}_{index}.png` (e.g., `0_001.png`, `1_002.png`, etc.).
2.  **Adjust Paths:** Modify the file paths in `main.rs` to point to your training and testing image directories:

    ```
    let mut files = get_filenames_in_dir("D:/Desktop/mnist/train"); // Change this line!
    // ...
    let mut test_files = get_filenames_in_dir("D:/Desktop/mnist/test"); // And this line!
    ```

    Replace `"D:/Desktop/mnist/train"` and `"D:/Desktop/mnist/test"` with the actual paths to your directories. **Important:** Use forward slashes (`/`) even on Windows.
3.  **Run the code:** Open your terminal, navigate to the project directory, and run:

    ```
    cargo run
    ```

    The AI will train and save the trained network to `network.json`.

### 3. Testing the AI with a Single Image

To test the AI with a single image, follow these steps:

1.  **Comment out training code:** Inside the `main()` function, comment out the entire training loop.
2.  **Call `forward_test()`:** Call the `forward_test()` function inside the `main()` function.
3.  **Adjust Image Path:** In the `forward_test()` function, change the `input.png` path to the image you want to test. Make sure the image is 28x28 pixels and grayscale.

    ```
    fn forward_test() {
        let mut network = Network::load("network.json");
        let parsed_image = parse_image("your_image.png".to_string()); // Change this line!
        let res = network.forward(parsed_image);

        visualize_output(res);
    }
    ```

4.  **Run the code:**

    ```
    cargo run
    ```

    The program will process the image and print the AI's guess.

### 4. Using Your Own Image

1.  **Image format:** Ensure your image is a 28x28 pixel grayscale image (PNG or JPG are recommended).
2.  **Image Path:** Make sure the path to your image is correctly specified in the `forward_test()` function as described above.

## Code Structure

*   `Network`: Represents the neural network.
*   `Layer`: Represents a layer in the network.
*   `Neuron`: Represents a neuron in a layer.
*   `parse_image()`: Loads and converts the image to a vector of f32 values (pixel brightness).
*   `forward()`: Performs a forward pass through the network.
*   `train()`: Trains the network using backpropagation.
*   `save()`/`load()`: Saves/loads the network to/from a JSON file.
