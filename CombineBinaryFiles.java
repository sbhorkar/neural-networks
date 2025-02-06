import java.io.*;

public class CombineBinaryFiles {
    public static void main(String[] args) {
        // Output file path
        String outputFile = "combined-training-blue.bin";
        
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(outputFile))) {
            // Process files in the specified order: 1-1, 2-1, ..., 5-1, 1-2, ..., 5-5
            for (int col = 1; col <= 5; col++) { // For columns 1 to 5
                for (int row = 1; row <= 5; row++) { // For rows 1 to 5
                    String fileName = String.format("%d-%d-cropped-final-blue-doubles.bin", row, col);
                    try (DataInputStream in = new DataInputStream(new FileInputStream(fileName))) {
                        System.out.println(fileName);
                        while (in.available() > 0) {
                            double data = in.readDouble();
                            out.writeDouble(data);
                        }
                    } catch (FileNotFoundException e) {
                        System.err.println("File not found: " + fileName);
                    } catch (IOException e) {
                        System.err.println("I/O Error for file: " + fileName);
                    }
                }
            }
            System.out.println("All files have been combined successfully into " + outputFile);
        } catch (IOException e) {
            System.err.println("Error writing to output file: " + e.getMessage());
        }
    }
}
