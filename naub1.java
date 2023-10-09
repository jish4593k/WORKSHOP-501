import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.types.UInt8;
import org.tensorflow.types.family.TNumber;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class AdvancedFaceDetectionJava {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static Session loadModel(String modelPath) {
        try (Graph graph = new Graph()) {
            byte[] graphDef = Files.readAllBytes(Paths.get(modelPath));
            graph.importGraphDef(graphDef);
            return new Session(graph);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String detectFacesInImage(String imagePath, Session session) {
        Mat image = Imgcodecs.imread(imagePath);
        Imgproc.resize(image, image, new Size(256, 256));
        image.convertTo(image, CvType.CV_32F, 1.0 / 255.0);

        Tensor<UInt8> inputTensor = Tensors.create(image);
        Tensor<?> outputTensor = session.runner()
                .feed("conv2d_input", inputTensor)
                .fetch("dense_1/Sigmoid")
                .run()
                .get(0);

        float[] predictions = new float[1];
        outputTensor.copyTo(predictions);

        if (predictions[0] > 0.5) {
            return "Лицо обнаружено";
        } else {
            return "Лицо не обнаружено";
        }
    }

    public static List<String> loadDataset(String datasetPath) {
        List<String> imagePaths = new ArrayList<>();
        try {
            Files.walk(Paths.get(datasetPath))
                    .filter(Files::isRegularFile)
                    .forEach(file -> imagePaths.add(file.toString()));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return imagePaths;
    }

    public static void trainModel(List<String> imagePaths, Session modelSession) {
        for (String imagePath : imagePaths) {
            String label = imagePath.contains("face") ? "1" : "0";

            Mat image = Imgcodecs.imread(imagePath);
            Imgproc.resize(image, image, new Size(256, 256));
            image.convertTo(image, CvType.CV_32F, 1.0 / 255.0);

            Tensor<UInt8> inputTensor = Tensors.create(image);
            Tensor<String> labelTensor = Tensors.create(label);

            //  If You want you can add code to train the model here using the input and label tensors
            
        }
    }

    public static double evaluateModel(List<String> testImagePaths, Session modelSession) {
        int correctPredictions = 0;
        int totalPredictions = 0;

        for (String imagePath : testImagePaths) {
            String label = imagePath.contains("face") ? "1" : "0";

            Mat image = Imgcodecs.imread(imagePath);
            Imgproc.resize(image, image, new Size(256, 256));
            image.convertTo(image, CvType.CV_32F, 1.0 / 255.0);

            Tensor<UInt8> inputTensor = Tensors.create(image);
            Tensor<String> labelTensor = Tensors.create(label);

            Tensor<Float> outputTensor = modelSession.runner()
                    .feed("conv2d_input", inputTensor)
                    .fetch("dense_1/Sigmoid")
                    .run()
                    .get(0)
                    .expect(Float.class);

            float[] prediction = new float[1];
            outputTensor.copyTo(prediction);

            int predictedLabel = (prediction[0] > 0.5) ? 1 : 0;
            int trueLabel = Integer.parseInt(label);

            if (predictedLabel == trueLabel) {
                correctPredictions++;
            }

            totalPredictions++;
        }

        return (double) correctPredictions / totalPredictions;
    }

    public static void main(String[] args) {
        String modelPath = "fdc.pb"; // Provide the path to the TensorFlow model
        Session modelSession = loadModel(modelPath);

        if (modelSession != null) {
            String imagePath = "123.png"; 
            String result = detectFacesInImage(imagePath, modelSession);
            System.out.println(result);

            String datasetPath = "dataset/"; 
            List<String> imagePaths = loadDataset(datasetPath);

            if (!imagePaths.isEmpty()) {
                trainModel(imagePaths, modelSession);
                System.out.println("Training completed.");

                String testDatasetPath = "test_dataset/";
                List<String> testImagePaths = loadDataset(testDatasetPath);

                if (!testImagePaths.isEmpty()) {
                    double accuracy = evaluateModel(testImagePaths, modelSession);
                    System.out.println("Model accuracy: " + accuracy);
                }
            }
        }
    }
}
