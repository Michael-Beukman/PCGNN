
package smb.lvl.analysis;
/**
 * Super Mario Bros. Level Visualizer.
 * 
 * The program reads levels in .txt-format and creates an PNG-image.
 * 
 * Set-up:
 * 1) Placer your levels (data) in the "read"-folder in a folder with 
 * a generator name.
 * 
 * 2) Set the "generatorName" and "fileNamePrefix" class members.
 * 
 * 3) In the main-method set "noOfLevels" to run incrementially 
 * covering all levels (0-X). 
 * 
 * Notes:
 * ... Currently it takes around 2 seconds per level to generate an image and 
 * save to file while running in an IDE.
 * 
 * @author Steve Dahlskog, Malmö University version: v0.2 (2015-04-06)
 * Version 0.2:
 * Treats sequences of 'p' as "Grassy knoll"-platforms and 
 * NOT as "Moving platforms".
 */

import java.io.*;
import java.awt.image.BufferedImage; 
import java.io.File; 
import java.io.IOException; 
import javax.imageio.ImageIO;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** 
 * 
 * Un-used imports in current version project:
 * import java.util.Date;
 * import java.text.SimpleDateFormat;
 * import java.util.Calendar;
 * import java.text.DateFormat;
 * import java.nio.charset.Charset;
 * import java.awt.Graphics2D; 
 * import java.io.FileInputStream; 
 * import java.util.Arrays;
 */

public class SMBLvlVisualizer {
    // SET-UP parameters
    static String path = "../levels"; // read level data from here
    static String generatorName = "original/renamed";
    static String fileNamePrefix = "new_lvl-";
    //
    
    // Class-members
    static char[][] level = new char[200][15];
    static char[][] level2 = new char[15][200];
    
    
    /**
     * Deprecated method
     * @param level
     * @param filetype
     * @param slice
     * @param whitespace
     * @return
     * @throws IOException
     * @throws Exception 
     */
    private static BufferedImage sliceimages(int level[], String filetype, char[] slice, boolean whitespace) throws IOException, Exception{
        // sliceimages(int level[], ...
        //private static void filetoimage(int level[], String filetype) throws IOException, Exception{
        int[] temp = new int[200];
        
        BufferedImage imgLib[] = new BufferedImage[29];
        String filename = "output_img/"; // Output
        String filenumber = "";
        File file;
        
        char wtspace[] = new char[] {'z','z','z','z','z','z','z','z','z','z','z','z','z','z','z'};
        char real[] = slice;
        
        File file0 = new File("tiles/sky.png");
        File file1 = new File("tiles/brick.png");
        File file2 = new File("tiles/ground.png");
        File file3 = new File("tiles/qm.png");
        File file4 = new File("tiles/powerup-qm.png");
        File file5 = new File("tiles/rock.png");
        File file6 = new File("tiles/goomba-in-the-sky.png");
        File file7 = new File("tiles/pipe.png");
        File file8 = new File("tiles/koopa.png");
        File file9 = new File("tiles/empty.png");
        File file10 = new File("tiles/piper.png");//File("tiles/piper.png");
        File file11 = new File("tiles/pipert.png");
        File file12 = new File("tiles/pipet.png");
        File file13 = new File("tiles/coin.png");
        File file14 = new File("tiles/moving-platform.png");
        File file15 = new File("tiles/red-koopa.png");
        File file16 = new File("tiles/koopawings.png");
        File file17 = new File("tiles/spring.png");
        File file18 = new File("tiles/lakitu.png");
        File file19 = new File("tiles/beetle.png");
        File file20 = new File("tiles/canon.png");
        File file21 = new File("tiles/canonbase.png");
        File file22 = new File("tiles/bridge.png");
        File file23 = new File("tiles/ham1.png");
        File file24 = new File("tiles/ham2.png");
        File file25 = new File("tiles/red-koopa-fly.png");
        File file26 = new File("island-center.png");
        File file27 = new File("island-left.png");
        File file28 = new File("island-right.png");
        File file29 = new File("island-base.png");
        
        imgLib[0] = ImageIO.read(file0); // = '0' - Sky
        imgLib[1] = ImageIO.read(file1); // = 'm' - Brick
        imgLib[2] = ImageIO.read(file2); // = 'g' - Ground
        imgLib[3] = ImageIO.read(file3); // = 'o' - ? Coin
        imgLib[4] = ImageIO.read(file4); // =  'w' - ? Powerup
        imgLib[5] = ImageIO.read(file5); // = 'r' - Rock
        imgLib[6] = ImageIO.read(file6); // = 'e' - Goomba
        imgLib[7] = ImageIO.read(file7); // = 't' - Pipe
        imgLib[8] = ImageIO.read(file8); // = 'k' - Koopa
        imgLib[9] = ImageIO.read(file9); // = '.' - empty
        imgLib[10] = ImageIO.read(file10); // = '.' - 
        imgLib[11] = ImageIO.read(file11); // = '.' - 
        imgLib[12] = ImageIO.read(file12); // = '.' - 
        imgLib[13] = ImageIO.read(file13); // = 'a' - coin free
        imgLib[14] = ImageIO.read(file14); // = 'p' - moving platform
        imgLib[15] = ImageIO.read(file15); // = 'd' - red koopa
        imgLib[16] = ImageIO.read(file16); // = 'K' - green wing koopa
        imgLib[17] = ImageIO.read(file17); // = 'Y' - spring
        imgLib[18] = ImageIO.read(file18); // = 'Q' - Lakitu
        imgLib[19] = ImageIO.read(file19); // = 'v'
        imgLib[20] = ImageIO.read(file20); // = 'c'  Canon
        imgLib[21] = ImageIO.read(file21); // = 'c' + '0'  Canonbase
        imgLib[22] = ImageIO.read(file22); // = 'P'
        imgLib[23] = ImageIO.read(file23); // = 'N' Hambros
        imgLib[24] = ImageIO.read(file24); // = 'N' + '0' Hambros
        imgLib[25] = ImageIO.read(file25); // = 'D' + red koopa wings
        imgLib[26] = ImageIO.read(file26); // = '?' island-center
        imgLib[27] = ImageIO.read(file27); // = '?' island-left
        imgLib[28] = ImageIO.read(file28); // = '?' island-right
        imgLib[29] = ImageIO.read(file29); // = '?' island-base
        
        
        int widthImg1 = imgLib[0].getWidth();
        int heightImg1 = imgLib[1].getHeight();

        
        BufferedImage img = new BufferedImage(
            widthImg1, // Final image will have width and height as
            heightImg1*15, // addition of widths and heights of the images we already have
            BufferedImage.TYPE_INT_RGB);
        
        boolean image1Drawn = false;
        
        for (int count = 0; count<slice.length ; count++){
            /*
            if (whitespace == true){
                slice = wtspace;
            }*/
            
            switch (slice[count]) {
            case 'r': image1Drawn = img.createGraphics().drawImage(imgLib[5], 0, heightImg1*count, null);
                    break;
            case 'e':  image1Drawn = img.createGraphics().drawImage(imgLib[6], 0, heightImg1*count, null);
                     break;
            case 'm':  image1Drawn = img.createGraphics().drawImage(imgLib[1], 0, heightImg1*count, null);
                     break;
            case 'l':  image1Drawn = img.createGraphics().drawImage(imgLib[1], 0, heightImg1*count, null);
                     break; // SOLVES SMB W2 L1 ladder to bonus area
            case 'g':  image1Drawn = img.createGraphics().drawImage(imgLib[2], 0, heightImg1*count, null);
                     break;
            case 'o':  image1Drawn = img.createGraphics().drawImage(imgLib[3], 0, heightImg1*count, null);
                     break;
            case 'w':  image1Drawn = img.createGraphics().drawImage(imgLib[4], 0, heightImg1*count, null);
                     break;
            case 'k':  image1Drawn = img.createGraphics().drawImage(imgLib[8], 0, heightImg1*count, null);
                     break;
            case 'a':  image1Drawn = img.createGraphics().drawImage(imgLib[13], 0, heightImg1*count, null);
                     break;
            case 'p':  image1Drawn = img.createGraphics().drawImage(imgLib[14], 0, heightImg1*count, null);
                     break;
            case 'P':  image1Drawn = img.createGraphics().drawImage(imgLib[22], 0, heightImg1*count, null);
                     break;
            case 'd':  image1Drawn = img.createGraphics().drawImage(imgLib[15], 0, heightImg1*count, null);
                     break;
            case 'K':  image1Drawn = img.createGraphics().drawImage(imgLib[16], 0, heightImg1*count, null);
                     break;
            case 'Q':  image1Drawn = img.createGraphics().drawImage(imgLib[18], 0, heightImg1*count, null);
                     break;
            case 'v':  image1Drawn = img.createGraphics().drawImage(imgLib[19], 0, heightImg1*count, null);
                     break; 
            case 'c':  image1Drawn = img.createGraphics().drawImage(imgLib[20], 0, heightImg1*count, null);
                     break; 
            case 'N':  image1Drawn = img.createGraphics().drawImage(imgLib[24], 0, heightImg1*count, null);   
                     break;
            case 'T':  
                       if(slice[count-1] == '0'||slice[count-1] == 'e'){
                           image1Drawn = img.createGraphics().drawImage(imgLib[11], 0, heightImg1*count, null);
                       }else{ 
                           image1Drawn = img.createGraphics().drawImage(imgLib[10], 0, heightImg1*count, null);
                       }
                     break;
            case 't':  
                       if(slice[count-1] == '0'||slice[count-1] == 'e'){
                           image1Drawn = img.createGraphics().drawImage(imgLib[12], 0, heightImg1*count, null);
                       }else{ 
                           image1Drawn = img.createGraphics().drawImage(imgLib[7], 0 ,heightImg1*count, null);
                       }
                     break;
            case 'Y':  image1Drawn = img.createGraphics().drawImage(imgLib[17], 0, heightImg1*count, null);
                // Check if not remove break;
                break;
            case 'D':  image1Drawn = img.createGraphics().drawImage(imgLib[25], 0, heightImg1*count, null);
                     break;
            case 'h':  image1Drawn = img.createGraphics().drawImage(imgLib[2], 0, heightImg1*count, null);//2
                     break;
                
            case '0':   
                        if(count < 13 && slice[count+1] == 'N'){
                            image1Drawn = img.createGraphics().drawImage(imgLib[23], 0, heightImg1*count, null);
                        }else if(count>1 && slice[count-1] =='c'){
                            image1Drawn = img.createGraphics().drawImage(imgLib[21], 0, heightImg1*count, null);
                        }else{
                            image1Drawn = img.createGraphics().drawImage(imgLib[0], 0, heightImg1*count, null);
                        }
                     break;
            default: image1Drawn = img.createGraphics().drawImage(imgLib[9], 0, heightImg1*count, null);
                     break;
        }
            
            
        }
        
        if(!image1Drawn) 
            System.out.println("Problems drawing first image"); //where we are placing image1 in final image

        boolean final_Image_drawing = true;
        if(!final_Image_drawing) 
            System.out.println("Problems drawing final image");
        
        
        return img;
    }
    
    /**
     * Method draws images for other method in vertical slices (1 tile wide and 15 tiles high)
     * @param level The level to be drawn
     * @param filetype
     * @param slice Which slice to draw
     * @param whitespace Currently not in use.
     * @return
     * @throws IOException
     * @throws Exception 
     */
    private static BufferedImage sliceimages(char level[][], String filetype, char[] slice, boolean whitespace) throws IOException, Exception{
        
        BufferedImage imgLib[] = new BufferedImage[30];
        //String filename = "Level_img/";
        //String filename = "results/data/";
        //String filenumber = "";
        //File file;
        
        char wtspace[] = new char[] {'z','z','z','z','z','z','z','z','z','z','z','z','z','z'};
        //char real[] = slice; //
        
        File file0 = new File("tiles/sky.png");
        File file1 = new File("tiles/brick.png");
        File file2 = new File("tiles/ground.png");
        File file3 = new File("tiles/qm.png");
        File file4 = new File("tiles/powerup-qm.png");
        File file5 = new File("tiles/rock.png");
        File file6 = new File("tiles/goomba-in-the-sky.png");
        File file7 = new File("tiles/pipe.png");
        File file8 = new File("tiles/koopa.png");
        File file9 = new File("tiles/empty.png");
        File file10 = new File("tiles/piper.png");//File("tiles/piper.png");
        File file11 = new File("tiles/pipert.png");
        File file12 = new File("tiles/pipet.png");
        File file13 = new File("tiles/coin.png");
        File file14 = new File("tiles/moving-platform.png");
        File file15 = new File("tiles/red-koopa.png");
        File file16 = new File("tiles/koopawings.png");
        File file17 = new File("tiles/spring.png");
        File file18 = new File("tiles/lakitu.png");
        File file19 = new File("tiles/beetle.png");
        File file20 = new File("tiles/canon.png");
        File file21 = new File("tiles/canonbase.png");
        File file22 = new File("tiles/bridge.png");
        File file23 = new File("tiles/ham1.png");
        File file24 = new File("tiles/ham2.png");
        File file25 = new File("tiles/red-koopa-fly.png");
        
        File file26 = new File("tiles/island-center.png");
        File file27 = new File("tiles/island-left.png");
        File file28 = new File("tiles/island-right.png");
        File file29 = new File("tiles/island-base.png");
        
        imgLib[0] = ImageIO.read(file0); // = '0' - Sky
        imgLib[1] = ImageIO.read(file1); // = 'm' - Brick
        imgLib[2] = ImageIO.read(file2); // = 'g' - Ground
        imgLib[3] = ImageIO.read(file3); // = 'o' - ? Coin
        imgLib[4] = ImageIO.read(file4); // =  'w' - ? Powerup
        imgLib[5] = ImageIO.read(file5); // = 'r' - Rock
        imgLib[6] = ImageIO.read(file6); // = 'e' - Goomba
        imgLib[7] = ImageIO.read(file7); // = 't' - Pipe
        imgLib[8] = ImageIO.read(file8); // = 'k' - Koopa
        imgLib[9] = ImageIO.read(file9); // = '.' - empty
        imgLib[10] = ImageIO.read(file10); // = '.' - 
        imgLib[11] = ImageIO.read(file11); // = '.' - 
        imgLib[12] = ImageIO.read(file12); // = '.' - 
        imgLib[13] = ImageIO.read(file13); // = 'a' - coin free
        imgLib[14] = ImageIO.read(file14); // = 'p' - moving platform
        imgLib[15] = ImageIO.read(file15); // = 'd' - red koopa
        imgLib[16] = ImageIO.read(file16); // = 'K' - green wing koopa
        imgLib[17] = ImageIO.read(file17); // = 'Y' - spring
        imgLib[18] = ImageIO.read(file18); // = 'Q' - Lakitu
        imgLib[19] = ImageIO.read(file19); // = 'v'
        imgLib[20] = ImageIO.read(file20); // = 'c'  Canon
        imgLib[21] = ImageIO.read(file21); // = 'c' + '0'  Canonbase
        imgLib[22] = ImageIO.read(file22); // = 'P'
        imgLib[23] = ImageIO.read(file23); // = 'N' Hambros
        imgLib[24] = ImageIO.read(file24); // = 'N' + '0' Hambros
        imgLib[25] = ImageIO.read(file25); // = 'D' + red koopa wings
        imgLib[26] = ImageIO.read(file26); // = '?' island-center
        imgLib[27] = ImageIO.read(file27); // = 'X' island-left
        imgLib[28] = ImageIO.read(file28); // = '?' island-right
        imgLib[29] = ImageIO.read(file29); // = '?' island-base
        
        int widthImg1 = imgLib[0].getWidth();
        int heightImg1 = imgLib[1].getHeight();

        BufferedImage img = new BufferedImage(
            widthImg1, // Final image will have width and height as
            heightImg1*15, // addition of widths and heights of the images we already have
            BufferedImage.TYPE_INT_RGB);
        
        boolean image1Drawn = false;
        
        for (int count = 0; count<slice.length ; count++){
            /*
            if (whitespace == true){
                slice = wtspace;
            }*/
            
            switch (slice[count]) {
            case 'r': image1Drawn = img.createGraphics().drawImage(imgLib[5], 0, heightImg1*count, null);
                    break;
            case 'e':  image1Drawn = img.createGraphics().drawImage(imgLib[6], 0, heightImg1*count, null);
                     break;
            case 'm':  image1Drawn = img.createGraphics().drawImage(imgLib[1], 0, heightImg1*count, null);
                     break;
            case 'l':  image1Drawn = img.createGraphics().drawImage(imgLib[1], 0, heightImg1*count, null);
                     break; // SOLVES SMB W2 L1 ladder to bonus area
            case 'g':  image1Drawn = img.createGraphics().drawImage(imgLib[2], 0, heightImg1*count, null);
                     break;
            case 'o':  image1Drawn = img.createGraphics().drawImage(imgLib[3], 0, heightImg1*count, null);
                     break;
            case 'w':  image1Drawn = img.createGraphics().drawImage(imgLib[4], 0, heightImg1*count, null);
                     break;
            case 'k':  image1Drawn = img.createGraphics().drawImage(imgLib[8], 0, heightImg1*count, null);
                     break;
            case 'a':  image1Drawn = img.createGraphics().drawImage(imgLib[13], 0, heightImg1*count, null);
                     break;
            case 'p':  image1Drawn = img.createGraphics().drawImage(imgLib[26], 0, heightImg1*count, null); //14
            // Problematic section, p is both a moving platform (tile img 14) and a grassy knoll platform (tile img 26).
            // Grass!
                     break;
            case 'P':  image1Drawn = img.createGraphics().drawImage(imgLib[22], 0, heightImg1*count, null);
                     break;
            case 'd':  image1Drawn = img.createGraphics().drawImage(imgLib[15], 0, heightImg1*count, null);
                     break;
            case 'K':  image1Drawn = img.createGraphics().drawImage(imgLib[16], 0, heightImg1*count, null);
                     break;
            case 'Q':  image1Drawn = img.createGraphics().drawImage(imgLib[18], 0, heightImg1*count, null);
                     break;
            case 'v':  image1Drawn = img.createGraphics().drawImage(imgLib[19], 0, heightImg1*count, null);
                     break; 
            case 'c':  image1Drawn = img.createGraphics().drawImage(imgLib[20], 0, heightImg1*count, null);
                     break; 
            case 'N':  image1Drawn = img.createGraphics().drawImage(imgLib[24], 0, heightImg1*count, null);   
                     break;
            case 'T':  
                       if(slice[count-1] == '0'||slice[count-1] == 'e'){
                           image1Drawn = img.createGraphics().drawImage(imgLib[11], 0, heightImg1*count, null);
                       }else{ 
                           image1Drawn = img.createGraphics().drawImage(imgLib[10], 0, heightImg1*count, null);
                       }
                     break;
            case 'F':  
                       if(slice[count-1] == '0'||slice[count-1] == 'e'){
                           image1Drawn = img.createGraphics().drawImage(imgLib[11], 0, heightImg1*count, null);
                       }else{ 
                           image1Drawn = img.createGraphics().drawImage(imgLib[10], 0, heightImg1*count, null);
                       }
                     break;
            case 't':  
                       if(slice[count-1] == '0'||slice[count-1] == 'e'){
                           image1Drawn = img.createGraphics().drawImage(imgLib[12], 0, heightImg1*count, null);
                       }else{ 
                           image1Drawn = img.createGraphics().drawImage(imgLib[7], 0 ,heightImg1*count, null);
                       }
                     break;
            case 'Y':  image1Drawn = img.createGraphics().drawImage(imgLib[17], 0, heightImg1*count, null);
                // Check if not remove break;
                break;
            case 'X': image1Drawn = img.createGraphics().drawImage(imgLib[27], 0, heightImg1*count, null);
                break;
            case 'x': image1Drawn = img.createGraphics().drawImage(imgLib[28], 0, heightImg1*count, null);
                break;
            case 'D':  image1Drawn = img.createGraphics().drawImage(imgLib[25], 0, heightImg1*count, null);
                     break;
            case 'h' : image1Drawn = img.createGraphics().drawImage(imgLib[29], 0, heightImg1*count, null); // Grass Base
                    break;
            case '0':   
                        if(count < 13 && slice[count+1] == 'N'){
                            image1Drawn = img.createGraphics().drawImage(imgLib[23], 0, heightImg1*count, null);
                        }else if(count>1 && slice[count-1] =='c'){
                            image1Drawn = img.createGraphics().drawImage(imgLib[21], 0, heightImg1*count, null);
                        }else{
                            image1Drawn = img.createGraphics().drawImage(imgLib[0], 0, heightImg1*count, null);
                        }
                     break;
            default: image1Drawn = img.createGraphics().drawImage(imgLib[9], 0, heightImg1*count, null);
                     break;
        }
            
            
        }
        
        if(!image1Drawn) 
            System.out.println("Problems drawing first image"); //where we are placing image1 in final image

        //if(!image2Drawn) 
        //  System.out.println("Problems drawing second image"); // image1 so both images will come in same level
        // horizontally
        //File final_image = new File("levels/xxx_" + seed + ".png"); // “png can also be used here”
        //File final_image = new File("slices/slice_" + filetype + "_"+ dateFormat.format(cal.getTime()) +".png"); // “png can also be used here”
        
        //boolean final_Image_drawing = ImageIO.write(img, "png", final_image); //if png is used, write “png” instead “jpeg”
        boolean final_Image_drawing = true;
        if(!final_Image_drawing){ 
            System.out.println("Problems drawing final image");
        }
        
        return img;
    }
    
    private static void filetoimage2(BufferedImage[] level, String ngramtype, boolean whitespace) throws IOException, Exception{
        
        BufferedImage imgLib[] = new BufferedImage[26];
        String filename;
        String filenumber;
        File file;
        
        int widthImg1 = level[0].getWidth();
        int heightImg1 = level[0].getHeight();
        
        BufferedImage img = new BufferedImage(
            widthImg1*level.length,//*100??? // Final image will have width and height as
            heightImg1, // addition of widths and heights of the images we already have
            BufferedImage.TYPE_INT_RGB);
        
        boolean image1Drawn = false;
        boolean image2Drawn = false;
        
        //image1Drawn = img.createGraphics().drawImage(imgLib[marioLevelGeneration[0][0]], 0, 0, null); // 0, 0 are the x and y positions
        image1Drawn = img.createGraphics().drawImage(level[0], 0, 0, null); // 0, 0 are the x and y positions
        
        for(int i = 1; i < level.length; i++){
            //image2Drawn = img.createGraphics().drawImage(imgLib[marioLevelGeneration[0][i]], widthImg1*i, 0, null); // here width is mentioned as width of
            //image2Drawn = img.createGraphics().drawImage(imgLib[temp[i]], widthImg1*i, 0, null); // here width is mentioned as width of   
            image2Drawn = img.createGraphics().drawImage(level[i], widthImg1*i, 0, null); // here width is mentioned as width of   
        }
        
        if(!image1Drawn) 
            System.out.println("Problems drawing first image"); //where we are placing image1 in final image

        if(!image2Drawn) 
          System.out.println("Problems drawing second image"); // image1 so both images will come in same level
        
        File final_image = new File("output_img/" + ngramtype +".png");
        boolean final_Image_drawing = ImageIO.write(img, "png", final_image); //if png is used, write “png” instead “jpeg”
        
        if(!final_Image_drawing) 
            System.out.println("Problems drawing final image");
    }
    
    private static void readFile(String fileName){
        int height, width;
        System.out.println("Level = " + fileName);

        BufferedReader reader = null;
        try{
            reader = new BufferedReader(new FileReader(fileName));
            String firstLine = reader.readLine();
            if(isCorrectHeader(firstLine)){
                String[] split = firstLine.split(";");
                height = new Integer(split[1].split("=")[1]);
                width = new Integer(split[0].split("=")[1]);
                
               level = new char[width][height]; 
                
               System.out.println("WIDTH/HEIGHT: " + width + " / " + height); 
                
                parseLevel(reader, height, width);
                //fixHillsAndPlatforms();
            } else {
                System.out.println("Incorrect file format. Exiting");
                reader.close();
                //return;
            }
        } catch (FileNotFoundException e){
            System.out.println("Unable to locate file.");
        } catch (IOException e) {
            System.out.println("Unable to read file.");
        } finally {
            try{
                reader.close();
            } catch (Exception e){
                System.out.println("Critical Exception: Not possible to close file. " + e.getLocalizedMessage());
            }
        }
    }
    
    /**
     * Currently returning TRUE for all possible String:s
     * @param line
     * @return boolean (true/false). 
     */
    private static boolean isCorrectHeader(String line){
        if(line.matches("HEIGHT=[0-9]+;WIDTH=[0-9]+$"))        {
            System.out.println("Pattern Matches");
            return true;
        } else {
            /**
             * Un-supported functionality. Planned for future version.
             */
            
            //System.out.println("Pattern does not match. Invalid file header.");
            //return false;
            return true;
        }
    } 
    
    private static void parseLevel(BufferedReader reader, int width,int height) throws IOException{
        for(int y = 0; y < height; y++){
            int x = 0;
            String line = reader.readLine();
            /**
             * Log-function - parsing all read lines.
             */
            //System.out.println(line);
            Pattern pattern = Pattern.compile("\\[[0-9a-zA-Z#]+\\]");
            Matcher matcher = pattern.matcher(line);

            while(matcher.find()) {
                if(x >= width){
                    System.out.println("Level width " + x);
                    System.out.println("Level data too long.");
                    return;
                }

                CharSequence seq = matcher.group().subSequence(1, matcher.group().length()-1);
                /*
                addLevelObjects(seq, x, y);
                */
                //System.out.println(seq);
                level[y][x] = seq.charAt(0);
                //System.out.println(level[y][x]);
                x++;
            }
        }
    }
    
    public static void main(String args[]){
        //Set up
        int noOfLevels = 1000;
        noOfLevels = 15; // Just for testing! Set this to the number of levels 
                        // you need to visualize
        //
        
        boolean drawGrass = true; // Added in v0.2
                
        for(int count = 0; count < noOfLevels; count++){
            readFile(path + "/" + generatorName + "/" + fileNamePrefix + count + ".txt");
            String out = "";
            
            
            
             /**
             * New to version 0.2
             * Due to data format weaknesses the visualizer can not see differnce of
             * Moving-platforms and Grassy knoll-platforms (see SMB. (1985).
             * 
             * Currently this is handled by reading the level representation and
             * replacing symbols here and there (i.e. long sequence of p should 
             * be *pppp* instead of pppppp.
             */
            
            for (int i = 0; i < 2; i++){ // level.length
                for (int j = 0; j < level[i].length; j++){
                    System.out.print(String.valueOf(level[i][j]));
                }
                System.out.println();
            }
            
            for (int i = 0; i < level.length; i++){
                for (int j = 0; j < level[i].length-1; j++){
                    if(level[i][j] == 'p' && level[i][j+1] == 'p' && drawGrass == true){
                        level[i][j] = 'X';
                        drawGrass = false;
                    }
                    if (level[i][j] == 'p' && level[i][j+1] != 'p' && j != level[i].length-1){
                        level[i][j] = 'x';
                        drawGrass = true;
                    }
                    /*
                    if(level[i][j] == 'p' && j == level[i].length-1){
                        level[i][j] = 'x';
                        drawGrass = true;
                        
                    }
                    /*
                    if (level[i][level[i].length-1] != 'x' && level[i][level[i].length-2] == 'p'){//&& level[i][level[i].length-2] == 'p'
                        level[i][(level[i].length-1)] = 'x';
                        level[i][(level[i].length-2)] = 'p';
                        drawGrass = false;
                    }/*
                    if (level[i].length-2 == 'X' && level[i].length-1 == 'p'){
                        level[i][(level[i].length-1)] = 'x';
                        //drawGrass = true;
                    }*/
                }
                drawGrass = true;
            }
            
            
            
            

            // This is used to flip the level char[] to a readable format
            level2 = new char[level[0].length][level.length];
            //Flip the level from line-reading-format to image-create-format.
            for (int x=0; x<level[0].length;x++){
                for (int y = 0; y<level.length;y++){
                    level2[x][y] = level[y][x];
                }
            }
            level = level2;
            
           
            

            BufferedImage generatedLevel[] = new BufferedImage[level.length];
            try{
                for(int u = 0; u < level.length; u++){
                    // Fill with slices
                    generatedLevel[u] = sliceimages(level, out, level[u], false);
                }
                //save file image
                filetoimage2(generatedLevel, fileNamePrefix + (count+1), false);
            } catch (Exception e){
                System.out.println("Something went wrong, while saving the slice-image. " + e.toString());
            }
        }
    }
    
}