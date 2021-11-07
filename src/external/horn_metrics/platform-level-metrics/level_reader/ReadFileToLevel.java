package dk.itu.mario.level;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import dk.itu.mario.MarioInterface.LevelInterface;
import dk.itu.mario.engine.sprites.SpriteTemplate;

public class ReadFileToLevel
{
	private Level level;
	private int theme;
	private boolean override;
	
	public Level readFileToLevel(String fileName)
	{
		readFile(fileName);
		
		return level;
	}
	
	private void readFile(String fileName)
	{
		
		BufferedReader reader = null;
		try
		{
			reader = new BufferedReader(new FileReader(fileName));
			String firstLine = reader.readLine();
			if(isCorrectHeader(firstLine))
			{
				String[] split = firstLine.split(";");
				if(split.length < 4)
				{
					level = new Level(new Integer(split[1].split("=")[1]), new Integer(split[0].split("=")[1]));
					theme = 0;
					override = true;
				}
				else
				{
					for (int i = 0; i < split.length; i++)
						System.out.println(i + " " + split[i]);
					level = new Level(new Integer(split[1].split("=")[1]), new Integer(split[0].split("=")[1]));
					theme = new Integer(split[2].split("=")[1]);
					override = new Boolean(split[3].split("=")[1]);
				}
				parseLevel(reader);
				fixHillsAndPlatforms();
				if(theme != LevelInterface.TYPE_OVERGROUND || override)
					fixWalls(level);
			}
			else
			{
				System.out.println("Incorrect file format. Exiting");
				level = null;
				reader.close();
				return;
			}
		} catch (FileNotFoundException e)
		{
			System.out.println("Unable to locate file.");
			e.printStackTrace();
		}
		catch (IOException e)
		{
			System.out.println("Unable to read file.");
			e.printStackTrace();
		}
		finally
		{
			try
			{
				reader.close();
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}
	}
	
	/*
	 * files should be written using this line of code for the first line:
	 * writer.write("HEIGHT=" + height + ";WIDTH=" + width + "\n");
	 */
	private boolean isCorrectHeader(String line)
	{
		System.out.println(line);
		if(line.matches("HEIGHT=[0-9]+;WIDTH=[0-9]+;THEME=[0-2]+;OVERRIDE=(false|true)$"))
		{
//			System.out.println("Pattern Matches");
			return true;
		}
		else if(line.matches("HEIGHT=[0-9]+;WIDTH=[0-9]+;"))
		{
			return true;
		}
		else if(line.matches("HEIGHT=[0-9]+;WIDTH=[0-9]+"))
		{
			return true;
		}
		else
		{
			System.out.println("Pattern does not match. Invalid file header.");
			return false;
		}
	}
	
	/*
	 * This function will read the 2D array from file and convert it to a Level
	 * using setSpriteTemplate and setBlock for each tile. Called after the 
	 * first line is validated and the level has been created with height and width.
	 */
	private void parseLevel(BufferedReader reader) throws IOException
	{
		for(int y=0; y<level.getHeight(); y++)
		{
			int x=0;
			String line = reader.readLine();
//			System.out.println(line);
			Pattern pattern = Pattern.compile("\\[[0-9a-zA-Z#]+\\]");
			Matcher matcher = pattern.matcher(line);
			while(matcher.find())
			{
				if(x>=level.getWidth())
				{
					System.out.println("Level data too long. x=" + x +  " level width=" + level.getWidth());
					level = null;
					return;
				}
				CharSequence seq = matcher.group().subSequence(1, matcher.group().length()-1);
				addLevelObjects(seq, x, y);
				x++;
			}
		}
	}
	
	private void addLevelObjects(CharSequence seq, int x, int y)
	{
		for(int i=0; i<seq.length(); i++)
		{
			char c = seq.charAt(i);
			switch (c)
			{
				case('0'):
					// ignore 0's
					break;
				case(LevelConstants.ground):
					addGround(x, y);
					break;
				case(LevelConstants.platform):
					addPlatform(x, y);
					break;
				case(LevelConstants.hill):
					addHill(x, y);
					break;
				case(LevelConstants.tube):
					addTube(x, y);
					break;
				case(LevelConstants.cannon_on_hill):
					addCannon(x, y);
					break;
				case(LevelConstants.small_tube_top):
					this.level.setBlock(x, y, LevelConstants.SMALL_TUBE_UP);
					break;
				case(LevelConstants.small_tube_middle):
					this.level.setBlock(x, y, LevelConstants.SMALL_TUBE_MID);
					break;
				case(LevelConstants.small_tube_down):
					this.level.setBlock(x, y, LevelConstants.SMALL_TUBE_DOWN);
					break;
				case(LevelConstants.log_bottom):
					this.level.setBlock(x, y, LevelConstants.LOG_BOTTOM);
					break;
				case(LevelConstants.log_middle):
					this.level.setBlock(x, y, LevelConstants.LOG_MIDDLE);
					break;
				case(LevelConstants.log_top):
					this.level.setBlock(x, y, LevelConstants.LOG_TOP);
					break;
				case(LevelConstants.wood_brick):
					this.level.setBlock(x, y, LevelConstants.WOOD_SQUARE);
					break;
				case(LevelConstants.blue_brick):
					this.level.setBlock(x, y, LevelConstants.BLUE_BRICK);
					break;
				case(LevelConstants.coin):
					this.level.setBlock(x, y, LevelConstants.COIN);
					break;
				case(LevelConstants.block_w_coin):
					// add block with coin
					this.level.setBlock(x, y, LevelConstants.BLOCK_COIN);
					break;
				case(LevelConstants.empty_block):
					// add rock with coin
					this.level.setBlock(x, y, LevelConstants.BLOCK_EMPTY);
					break;
				case(LevelConstants.block_power):
					// add block power
					this.level.setBlock(x, y, LevelConstants.BLOCK_POWERUP);
					break;
				case(LevelConstants.rock):
					// add empty rock
					this.level.setBlock(x, y, LevelConstants.ROCK);
					break;
				case(LevelConstants.goomba):
					// add goomba
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.GOOMPA, false));
					break;
				case(LevelConstants.goomba_wings):
					// add goomba wings
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.GOOMPA, true));
					break;
				case(LevelConstants.red_turtle):
					// add red turtle
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.RED_TURTLE, false));
					break;
				case(LevelConstants.red_turtle_wings):
					// add red turtle with wings
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.RED_TURTLE, true));
					break;
				case(LevelConstants.green_turtle):
					// add green turtle
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.GREEN_TURTLE, false));
					break;
				case(LevelConstants.green_turtle_wings):
					// add green turtle with wings
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.GREEN_TURTLE, true));
					break;
				case(LevelConstants.chomp_flower):
					// add chomp flower
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.CHOMP_FLOWER, false));
					break;
				case(LevelConstants.jump_flower):
					// add jump flower
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.JUMP_FLOWER, false));
					break;
				case(LevelConstants.cannon):
					// add cannon
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.CANNON_BALL, false));
					break;
				case(LevelConstants.armored_turtle):
					// add cannon
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.ARMORED_TURTLE, false));
					break;
				case(LevelConstants.armored_turtle_wings):
					// add cannon
					this.level.setSpriteTemplate(x, y, new SpriteTemplate(SpriteTemplate.ARMORED_TURTLE, true));
					break;
				case(LevelConstants.block_hidden_coin):
					// add hidden block with coin
					this.level.setBlock(x, y, LevelConstants.BLOCK_HIDDEN_COIN);
					break;
				case(LevelConstants.exit):
					// add level exit
					this.level.xExit=x;
					this.level.yExit=y;
					break;
				default:
					System.out.println("Unrecognized level object...skipping: " + c);
					break;
			}
		}
	}
	
	private void addGround(int x, int y)
	{
		this.level.setBlock(x, y, LevelConstants.GROUND);
	}
	
	private void addPlatform(int x, int y)
	{
		// set as top for now, will fix corners later
		this.level.setBlock(x, y, LevelConstants.HILL_TOP);
	}
	
	private void addHill(int x, int y)
	{
		// set as fill for now, will fix later
		this.level.setBlock(x, y, LevelConstants.HILL_FILL);
	}
	
	// this method assumes working from (0,0) to (n,n), x first then y
	private void addTube(int x, int y)
	{
		// tube top right
		if(this.level.getBlock(x-1, y)==LevelConstants.TUBE_TOP_LEFT)
			this.level.setBlock(x, y, LevelConstants.TUBE_TOP_RIGHT);
		// tube side right
		else if(this.level.getBlock(x-1, y)==LevelConstants.TUBE_SIDE_LEFT)
			this.level.setBlock(x, y, LevelConstants.TUBE_SIDE_RIGHT);
		
		// must be on the left side of the tube
		// tube side left
		else if(this.level.getBlock(x, y-1)==LevelConstants.TUBE_TOP_LEFT || this.level.getBlock(x, y-1)==LevelConstants.TUBE_SIDE_LEFT)
			this.level.setBlock(x, y, LevelConstants.TUBE_SIDE_LEFT);
		// nothing left so must be a tube top let
		else 
			this.level.setBlock(x, y, LevelConstants.TUBE_TOP_LEFT);
	}
	
	// this method assumes working from (0,0) to (n,n), x first then y
	private void addCannon(int x, int y)
	{
		System.out.println("ADDING CANNON");
		if(this.level.getBlock(x, y-1)!=LevelConstants.CANNON_TOP
				&& this.level.getBlock(x, y-1)!=LevelConstants.CANNON_MIDDLE
				&& this.level.getBlock(x, y-1)!=LevelConstants.CANNON_BOTTOM)
			this.level.setBlock(x, y, LevelConstants.CANNON_TOP);
		else if(this.level.getBlock(x, y-1)==LevelConstants.CANNON_TOP)
			this.level.setBlock(x, y, LevelConstants.CANNON_MIDDLE);
		else
			this.level.setBlock(x, y, LevelConstants.CANNON_BOTTOM);
	}
	
	// pass over level once to fix corners of platforms
	private void fixHillsAndPlatforms()
	{
		for(int y=0; y<this.level.getHeight(); y++)
		{
			for(int x=0; x<this.level.getWidth(); x++)
			{
				if(this.level.getBlock(x, y)==LevelConstants.HILL_TOP || this.level.getBlock(x, y)==LevelConstants.HILL_FILL)
				{
					// platform top
					if(!isHill(this.level.getBlock(x, y-1)))
					{
						if((isHill(this.level.getBlock(x-1, y)) || isTube(this.level.getBlock(x-1, y)))
								&& (!isHill(this.level.getBlock(x+1, y)) && !isTube(this.level.getBlock(x+1, y))))
						{
							// platform top right
							if(isHill(this.level.getBlock(x, y+1)) && !isHill(this.level.getBlock(x-1, y+1)))
								this.level.setBlock(x, y, LevelConstants.RIGHT_UP_GRASS_EDGE);
							else if(y==this.level.getHeight()-1)
								this.level.setBlock(x, y, LevelConstants.RIGHT_UP_GRASS_EDGE);
							else
								this.level.setBlock(x, y, LevelConstants.HILL_TOP_RIGHT);
						}
						else if((!isHill(this.level.getBlock(x-1, y)) && !isTube(this.level.getBlock(x-1, y)))
								&& (isHill(this.level.getBlock(x+1, y)) || isTube(this.level.getBlock(x+1, y))))
						{
							// platform top left
							if(isHill(this.level.getBlock(x, y+1)) && !isHill(this.level.getBlock(x+1, y+1)))
								this.level.setBlock(x, y, LevelConstants.LEFT_UP_GRASS_EDGE);
							else if(y==this.level.getHeight()-1)
								this.level.setBlock(x, y, LevelConstants.LEFT_UP_GRASS_EDGE);
							else
								this.level.setBlock(x, y, LevelConstants.HILL_TOP_LEFT);
						}
						else
						{
							// already a hill top so continue
						}
					}
					// hill bottom
					else if(!isHill(this.level.getBlock(x, y+1)) && this.level.getBlock(x, y+1) != LevelConstants.GROUND)
					{
						// hill bottom right
						if((isHill(this.level.getBlock(x-1, y)) || isTube(this.level.getBlock(x-1, y)))
								&& (!isHill(this.level.getBlock(x+1, y)) && !isTube(this.level.getBlock(x+1, y))))
						{
							this.level.setBlock(x, y, LevelConstants.HILL_BOTTOM_LEFT);
						}
						// hill bottom left
						else if((!isHill(this.level.getBlock(x-1, y)) && !isTube(this.level.getBlock(x-1, y)))
								&& (isHill(this.level.getBlock(x+1, y)) || isTube(this.level.getBlock(x+1, y))))
						{
							this.level.setBlock(x, y, LevelConstants.HILL_BOTTOM_RIGHT);
						}
						// hill bottom
						else
						{
							this.level.setBlock(x, y, LevelConstants.HILL_BOTTOM);
						}
					}
					// bottom left in
					else if((isHill(this.level.getBlock(x, y+1)) || this.level.getBlock(x, y+1) == LevelConstants.GROUND)
							&& (isHill(this.level.getBlock(x-1, y)) || this.level.getBlock(x-1, y) == LevelConstants.GROUND)
							&& (isHill(this.level.getBlock(x, y-1)) || this.level.getBlock(x, y-1) == LevelConstants.GROUND)
							&& (!isHill(this.level.getBlock(x-1, y+1)) && this.level.getBlock(x-1, y+1) != LevelConstants.GROUND))
					{
						this.level.setBlock(x, y, LevelConstants.HILL_BOTTOM_LEFT_IN);
					}
					// bottom right in
					else if((isHill(this.level.getBlock(x, y+1)) || this.level.getBlock(x, y+1) == LevelConstants.GROUND)
							&& (isHill(this.level.getBlock(x+1, y)) || this.level.getBlock(x+1, y) == LevelConstants.GROUND)
							&& (isHill(this.level.getBlock(x, y-1)) || this.level.getBlock(x, y-1) == LevelConstants.GROUND)
							&& (!isHill(this.level.getBlock(x+1, y+1)) && this.level.getBlock(x+1, y+1) != LevelConstants.GROUND))
					{
						this.level.setBlock(x, y, LevelConstants.HILL_BOTTOM_RIGHT_IN);
					}
					// platform right
					else if(!isHill(this.level.getBlock(x+1, y)) && !isTube(this.level.getBlock(x+1, y))
							&& this.level.getBlock(x+1, y)!=LevelConstants.GROUND && !isHill(this.level.getBlock(x-1, y)))
					{
						this.level.setBlock(x, y, LevelConstants.RIGHT_GRASS_EDGE);
//						if(isHill(this.level.getBlock(x, y-1)))
//						{
//							if(isHill(this.level.getBlock(x, y+1)))
//								this.level.setBlock(x, y, LevelConstants.HILL_RIGHT);
//							else if(isHill(this.level.getBlock(x+1, y)))
//								this.level.setBlock(x, y, LevelConstants.RIGHT_POCKET_GRASS);
//							else
//								// this is an odd case, just using hill right for now
//								this.level.setBlock(x, y, LevelConstants.HILL_RIGHT);
//						}
					}
					
					// platform left
					else if(!isHill(this.level.getBlock(x-1, y)) && !isTube(this.level.getBlock(x-1, y))
							&& this.level.getBlock(x-1, y)!=LevelConstants.GROUND && !isHill(this.level.getBlock(x+1, y)))
					{
						this.level.setBlock(x, y, LevelConstants.LEFT_GRASS_EDGE);
//						System.out.println("setting left edge: x" + x + " y" + y);
//						if(isHill(this.level.getBlock(x, y-1)))
//						{
//							if(isHill(this.level.getBlock(x, y+1)))
//								this.level.setBlock(x, y, LevelConstants.HILL_LEFT);
//							else if(isHill(this.level.getBlock(x+1, y)))
//								this.level.setBlock(x, y, LevelConstants.LEFT_POCKET_GRASS);
//							else
//								// this is an odd case, just using hill right for now
//								this.level.setBlock(x, y, LevelConstants.HILL_LEFT);
//						}
					}
					else if(!isHill(this.level.getBlock(x-1, y)) && !isTube(this.level.getBlock(x-1, y)))
					{
						this.level.setBlock(x, y, LevelConstants.HILL_TOP_LEFT);
					}
					else if(!isHill(this.level.getBlock(x+1, y)) && !isTube(this.level.getBlock(x+1, y)))
					{
						this.level.setBlock(x, y, LevelConstants.HILL_TOP_RIGHT);
					}
					
					// pocket
					if(isHill(this.level.getBlock(x-1, y))
							&& isHill(this.level.getBlock(x, y-1))
							&& !isHill(this.level.getBlock(x+1, y))
							&& !isHill(this.level.getBlock(x-1, y-1)))
					{
						this.level.setBlock(x, y, LevelConstants.RIGHT_POCKET_GRASS);
//						System.out.println("setting right pocket edge: x" + x + " y" + y);
					}
					// pocket
					if(isHill(this.level.getBlock(x+1, y))
							&& isHill(this.level.getBlock(x, y-1))
							&& !isHill(this.level.getBlock(x-1, y))
							&& !isHill(this.level.getBlock(x+1, y-1)))
					{
						this.level.setBlock(x, y, LevelConstants.LEFT_POCKET_GRASS);
//						System.out.println("setting left pocket: x" + x + " y" + y);
					}
				}
				
				/*
				// hills... for now, just doing the same thing as platforms.
				if(this.level.getBlock(x, y)==LevelConstants.HILL_FILL)
				{					// platform top
					if(!isHill(this.level.getBlock(x, y-1)))
					{
						if((isHill(this.level.getBlock(x-1, y)) || isTube(this.level.getBlock(x-1, y)))
								&& (!isHill(this.level.getBlock(x+1, y)) && !isTube(this.level.getBlock(x+1, y))))
						{
							// platform top right
							if(isHill(this.level.getBlock(x, y+1)))
								this.level.setBlock(x, y, LevelConstants.RIGHT_UP_GRASS_EDGE);
							else
								this.level.setBlock(x, y, LevelConstants.HILL_TOP_RIGHT);
						}
						else if((!isHill(this.level.getBlock(x-1, y)) && !isTube(this.level.getBlock(x-1, y)))
								&& (isHill(this.level.getBlock(x+1, y)) || isTube(this.level.getBlock(x+1, y))))
						{
							// platform top left
							if(isHill(this.level.getBlock(x, y+1)))
								this.level.setBlock(x, y, LevelConstants.LEFT_UP_GRASS_EDGE);
							else
								this.level.setBlock(x, y, LevelConstants.HILL_TOP_LEFT);
						}
						else
						{
							// already a hill top so continue
						}
					}
					
					// platform right
					else if(!isHill(this.level.getBlock(x+1, y)) && !isTube(this.level.getBlock(x+1, y)))
					{
						if(isHill(this.level.getBlock(x, y-1)))
						{
							if(isHill(this.level.getBlock(x, y+1)))
								this.level.setBlock(x, y, LevelConstants.HILL_RIGHT);
							else if(isHill(this.level.getBlock(x+1, y)))
								this.level.setBlock(x, y, LevelConstants.RIGHT_POCKET_GRASS);
							else
								// this is an odd case, just using hill right for now
								this.level.setBlock(x, y, LevelConstants.HILL_RIGHT);
						}
					}
					
					// platform left
					else if(!isHill(this.level.getBlock(x-1, y)) && !isTube(this.level.getBlock(x-1, y)))
					{
						if(isHill(this.level.getBlock(x, y-1)))
						{
							if(isHill(this.level.getBlock(x, y+1)))
								this.level.setBlock(x, y, LevelConstants.HILL_LEFT);
							else if(isHill(this.level.getBlock(x+1, y)))
								this.level.setBlock(x, y, LevelConstants.LEFT_POCKET_GRASS);
							else
								// this is an odd case, just using hill right for now
								this.level.setBlock(x, y, LevelConstants.HILL_LEFT);
						}
					}
				}
				*/
			}
		}
	}
	
	private boolean isTube(byte b)
	{
		if(b==LevelConstants.TUBE_SIDE_LEFT
				|| b==LevelConstants.TUBE_SIDE_RIGHT
				|| b==LevelConstants.TUBE_TOP_LEFT
				|| b==LevelConstants.TUBE_TOP_RIGHT
				|| b==LevelConstants.SMALL_TUBE_UP
				|| b==LevelConstants.SMALL_TUBE_MID
				|| b==LevelConstants.SMALL_TUBE_DOWN)
			return true;
		return false;
	}
	
	private boolean isHill(byte b)
	{
		if(b==LevelConstants.HILL_FILL
				|| b==LevelConstants.HILL_LEFT
				|| b==LevelConstants.HILL_RIGHT
				|| b==LevelConstants.HILL_TOP
				|| b==LevelConstants.HILL_TOP_LEFT
				|| b==LevelConstants.HILL_TOP_LEFT_IN
				|| b==LevelConstants.HILL_TOP_RIGHT
				|| b==LevelConstants.HILL_TOP_RIGHT_IN
				|| b==LevelConstants.LEFT_GRASS_EDGE
				|| b==LevelConstants.LEFT_POCKET_GRASS
				|| b==LevelConstants.LEFT_UP_GRASS_EDGE
				|| b==LevelConstants.RIGHT_GRASS_EDGE
				|| b==LevelConstants.RIGHT_POCKET_GRASS
				|| b==LevelConstants.RIGHT_UP_GRASS_EDGE
				|| b==LevelConstants.HILL_BOTTOM
				|| b==LevelConstants.HILL_BOTTOM_LEFT
				|| b==LevelConstants.HILL_BOTTOM_LEFT_IN
				|| b==LevelConstants.HILL_BOTTOM_RIGHT
				|| b==LevelConstants.HILL_BOTTOM_RIGHT_IN)
			return true;
		return false;
	}
	
	private boolean isGroundOrEmpty(byte b)
	{
		if(b==0 || b==LevelConstants.GROUND)
			return true;
		return false;
	}
	
	private void fixWalls(Level level)
	{
		int width = level.getWidth();
		int height = level.getHeight();
		
		boolean[][] blockMap = new boolean[width + 1][height + 1];
		for (int x = 0; x < width + 1; x++)
		{
			for (int y = 0; y < height + 1; y++)
			{
				int blocks = 0;
				for (int xx = x - 1; xx < x + 1; xx++)
				{
					for (int yy = y - 1; yy < y + 1; yy++)
					{
						if (level.getBlockCapped(xx, yy) == (byte) (1 + 9 * 16))
							blocks++;
					}
				}
				blockMap[x][y] = blocks == 4;
			}
		}
		blockify(level, blockMap, width + 1, height + 1);
	}

	private void blockify(Level level, boolean[][] blocks, int width, int height)
	{
		int to = 0;
		if (theme == Level.TYPE_CASTLE)
		{
			to = 4 * 2;
		} else if (theme == Level.TYPE_UNDERGROUND)
		{
			to = 4 * 3;
		}

		boolean[][] b = new boolean[2][2];
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				for (int xx = x; xx <= x + 1; xx++)
				{
					for (int yy = y; yy <= y + 1; yy++)
					{
						int _xx = xx;
						int _yy = yy;
						if (_xx < 0)
							_xx = 0;
						if (_yy < 0)
							_yy = 0;
						if (_xx > width - 1)
							_xx = width - 1;
						if (_yy > height - 1)
							_yy = height - 1;
						b[xx - x][yy - y] = blocks[_xx][_yy];
					}
				}

				if (b[0][0] == b[1][0] && b[0][1] == b[1][1])
				{
					if (b[0][0] == b[0][1])
					{
						if (b[0][0])
						{
							level.setBlock(x, y, (byte) (1 + 9 * 16 + to));
						} else
						{
							// KEEP OLD BLOCK!
						}
					} else
					{
						if (b[0][0])
						{
							level.setBlock(x, y, (byte) (1 + 10 * 16 + to));
						} else
						{
							level.setBlock(x, y, (byte) (1 + 8 * 16 + to));
						}
					}
				} else if (b[0][0] == b[0][1] && b[1][0] == b[1][1])
				{
					if (b[0][0])
					{
						level.setBlock(x, y, (byte) (2 + 9 * 16 + to));
					} else
					{
						level.setBlock(x, y, (byte) (0 + 9 * 16 + to));
					}
				} else if (b[0][0] == b[1][1] && b[0][1] == b[1][0])
				{
					level.setBlock(x, y, (byte) (1 + 9 * 16 + to));
				} else if (b[0][0] == b[1][0])
				{
					if (b[0][0])
					{
						if (b[0][1])
						{
							level.setBlock(x, y, (byte) (3 + 10 * 16 + to));
						} else
						{
							level.setBlock(x, y, (byte) (3 + 11 * 16 + to));
						}
					} else
					{
						if (b[0][1])
						{
							level.setBlock(x, y, (byte) (2 + 8 * 16 + to));
						} else
						{
							level.setBlock(x, y, (byte) (0 + 8 * 16 + to));
						}
					}
				} else if (b[0][1] == b[1][1])
				{
					if (b[0][1])
					{
						if (b[0][0])
						{
							level.setBlock(x, y, (byte) (3 + 9 * 16 + to));
						} else
						{
							level.setBlock(x, y, (byte) (3 + 8 * 16 + to));
						}
					} else
					{
						if (b[0][0])
						{
							level.setBlock(x, y, (byte) (2 + 10 * 16 + to));
						} else
						{
							level.setBlock(x, y, (byte) (0 + 10 * 16 + to));
						}
					}
				} else
				{
					level.setBlock(x, y, (byte) (0 + 1 * 16 + to));
				}
			}
		}
	}
}
