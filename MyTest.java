import java.io.*;

public class MyTest
   {
   public static final int RED   = 0x00FF0000;
   public static final int GREEN = 0x0000FF00;
   public static final int BLUE  = 0x000000FF;

   public static final int BLACK  = 0x00000000;
   public static final int WHITE  = 0x00FFFFFF;

   public static final int R = 1001;
   public static final int C = 1001;

   public static void main(String[] args)
      {
        int[][] bmp = new int[R][C];

        String inputName = "";
        String outputName = "";

        if (args.length > 0)
            {
                inputName = args[0];
                outputName = args[1];
            }

        try
        {
            FileInputStream stream = new FileInputStream(new File(inputName));
            DataInputStream data = new DataInputStream(stream);

            for (int row = 0; row < R; row++)
            {
                for (int column = 0; column < C; column++)
                {
                    bmp[row][column] = data.readByte();
                }
            }
        }
        catch (Exception e)
        {
            System.out.println("Exception: " + e);
        }

        PelArray pixels = new PelArray(bmp);
        System.out.printf("Image Width %d\n", pixels.getWidth());
        System.out.printf("Image Height %d\n\n", pixels.getHeight());
        PelArray pixels2 = new PelArray(bmp);        
        PelArray pixels3 = pixels2.grayScaleImage();       // Convert the image to gray
        PelArray blueOnly = pixels3.oneColorImage(pixels3.BLUE);   // Mask all but the least significant
        System.out.println("One's Complement Test");
        PelArray pix10 = blueOnly.onesComplimentImage();
        pix10 = pix10.scale(100, 100);
        int[][] fixed = pix10.getPelArray();
        System.out.println(fixed.length + " " + fixed[0].length);

        try
        {
            FileOutputStream output = new FileOutputStream(new File(outputName));
            DataOutputStream data = new DataOutputStream(output);

            for (int row = 0; row < fixed.length; row++)
            {
                for (int column = 0; column < fixed[0].length; column++)
                {
                    data.writeByte(fixed[row][column]);
                }
            }
        }
        catch (Exception e)
        {
            System.out.println("Exception: " + e);
        }
    }
}
