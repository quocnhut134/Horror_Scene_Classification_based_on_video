import io
import csv
import numpy as np

YAMNET_CLASS_MAP_CSV = """
mid,display_name,description
/m/09cpr,Speech,Speech
/m/09x0r,Child speech,Child speech
/m/02zso,Conversation,Conversation
/m/0dzql,Male speech,Male speech
/m/0kxll,Female speech,Female speech
/m/0gx6g,Singing,Singing
/m/02sqy,Whistling,Whistling
/m/04szw,Scream,Scream
/m/0gxvj,Narration,Narration
/m/0fxh6,Speech synthesizer,Speech synthesizer
/m/0gf28,Shout,Shout
/m/0hkj6,Whisper,Whisper
/m/0chx_r,Stutter,Stutter
/m/0t_z4,Laughter,Laughter
/m/014z_1,Chuckle,Chuckle
/m/0glj9,Giggle,Giggle
/m/0j2dx,Snicker,Snicker
/m/0xghr,Gasp,Gasp
/m/07p65g6,Burping,Burping
/m/01c48z,Fart,Fart
/m/014c5w,Cough,Cough
/m/0f3_l,Sneeze,Sneeze
/m/07gcl2,Snoring,Snoring
/m/08p55g,Breathing,Breathing
/m/02mk9,Wheeze,Wheeze
/m/01q393,Grumble,Grumble
/m/0dzct,Humming,Humming
/m/07qg2,Throat clearing,Throat clearing
/m/0g5_q,Crying,Crying
/m/03_z0t,Baby crying,Baby crying
/m/01y_3n,Whining,Whining
/m/0dvs_3,Sobbing,Sobbing
/m/04rf8,Sigh,Sigh
/m/0hgs8w,Groan,Groan
/m/0g29p,Gasp,Gasp
/m/0j_6v,Panting,Panting
/m/046z3j,Yawn,Yawn
/m/0_jb,Whisper,Whisper
/m/07m6w,Muttering,Muttering
/m/0805y8,Rumbling,Rumbling
/m/07ppk7m,Sniff,Sniff
/m/07j2l,Sigh,Sigh
/m/0brd_5,Whimpering,Whimpering
/m/07r1d,Chewing,Chewing
/m/0dv7_2,Gargling,Gargling
/m/0f81d,Dripping,Dripping
/m/0j44d,Slurping,Slurping
/m/0hgr4,Stomach rumble,Stomach rumble
/m/01q4x1,Burping,Burping
/m/06d2h,Fart,Fart
/m/0gjq7,Hiccup,Hiccup
/m/03b_9c,Snoring,Snoring
/m/0j540g,Coughing,Coughing
/m/0g5b_,Sneeze,Sneeze
/m/012l68,Humming,Humming
/m/0dxfb,Whistling,Whistling
/m/01hm0y,Breathing,Breathing
/m/0284z,Speech,Speech
/m/0c4gh,Music,Music
/m/0f844,Electronic music,Electronic music
/m/0gj2j,Pop music,Pop music
/m/0gf2w,Rock music,Rock music
/m/0hfnj,Hip hop music,Hip hop music
/m/0kbp_w,Jazz,Jazz
/m/0gg_8,Classical music,Classical music
/m/04xp5,Dance music,Dance music
/m/02mfv,Folk music,Folk music
/m/07pj1w,Country music,Country music
/m/0h557,Blues,Blues
/m/02rlv,Soul music,Soul music
/m/0g6z7,Reggae,Reggae
/m/0h053,Funk,Funk
/m/0fxh6,Disco,Disco
/m/0h2y4,Gospel music,Gospel music
/m/0gl8z,Soundtrack,Soundtrack
/m/0gwz_m,Spoken word,Spoken word
/m/0gf18,Acoustic music,Acoustic music
/m/0gj54,Instrumental music,Instrumental music
/m/0j3z1,Orchestra,Orchestra
/m/0gf50,Symphony,Symphony
/m/032s1,Opera,Opera
/m/0g7r4,Musical theatre,Musical theatre
/m/0h07p,Carnatic music,Carnatic music
/m/0h6gt,Gamelan,Gamelan
/m/0gj0d,Acoustic guitar,Acoustic guitar
/m/04sz7,Electric guitar,Electric guitar
/m/04x4x,Bass guitar,Bass guitar
/m/03p1r,Piano,Piano
/m/0gj0m,Keyboard,Keyboard
/m/0199g,Violin,Violin
/m/04_sv,Cello,Cello
/m/02rk6,Double bass,Double bass
/m/025_j,Harp,Harp
/m/0j0v2,Flute,Flute
/m/0gk6x,Clarinet,Clarinet
/m/0gh7f,Saxophone,Saxophone
/m/0gp20,Trumpet,Trumpet
/m/0gkxw,Trombone,Trombone
/m/0gkw7,French horn,French horn
/m/0gp9c,Tuba,Tuba
/m/0h09m,Accordion,Accordion
/m/01dclc,Bagpipes,Bagpipes
/m/0f80l,Didgeridoo,Didgeridoo
/m/0j0zb,Oboe,Oboe
/m/0gh2n,Bassoon,Bassoon
/m/0h2j3,Harmonica,Harmonica
/m/0j3p3,Percussion,Percussion
/m/0gj6d,Drum kit,Drum kit
/m/0gl0j,Cymbal,Cymbal
/m/0gj8w,Snare drum,Snare drum
/m/0h07v,Tabla,Tabla
/m/0gl9j,Tambourine,Tambourine
/m/0gh9f,Bongo,Bongo
/m/0g008,Maraca,Maraca
/m/0gvz2,Gong,Gong
/m/0gkjm,Bell,Bell
/m/0gf2l,Xylophone,Xylophone
/m/0j3x9,Glockenspiel,Glockenspiel
/m/0gk7l,Chimes,Chimes
/m/02mfw,Whistle,Whistle
/m/0gf2t,Siren,Siren
/m/0gf2s,Alarm,Alarm
/m/0gj46,Clock,Clock
/m/0j072,Tick-tock,Tick-tock
/m/0gl00,Telephone bell ringing,Telephone bell ringing
/m/0gj0n,Ringtone,Ringtone
/m/0gf3g,Buzzer,Buzzer
/m/0gj0l,Chime,Chime
/m/0gjy7,Ding,Ding
/m/0gf2q,Electronic bell,Electronic bell
/m/0gjsz,Handbell,Handbell
/m/0gj9p,Jingle,Jingle
/m/0gj47,Clang,Clang
/m/03cxv,Bicycle bell,Bicycle bell
/m/02p0w,Mechanical fan,Mechanical fan
/m/0dz_l,Air conditioning,Air conditioning
/m/0gj3t,Hum,Hum
/m/0gjk4,Whir,Whir
/m/0h15h,Engine,Engine
/m/0gxk3,Vehicle,Vehicle
/m/0155b,Car,Car
/m/0kbp_w,Motor vehicle (road),Motor vehicle (road)
/m/0h5k4,Truck,Truck
/m/0h2n6,Motorcycle,Motorcycle
/m/0gjf3,Bicycle,Bicycle
/m/0gzx8,Aircraft,Aircraft
/m/01j3zr,Fixed-wing aircraft,Fixed-wing aircraft
/m/09rvn,Helicopter,Helicopter
/m/0h36v,Train,Train
/m/0h4_k,Sailboat,Sailboat
/m/0gylm,Bus,Bus
/m/0j05v,Fire engine,Fire engine
/m/0j3y2,Ambulance,Ambulance
/m/07rwj,Police car (siren),Police car (siren)
/m/01jg02,Motorboat,Motorboat
/m/0gjq1,Ship,Ship
/m/0c326,Boat,Boat
/m/080j_q,Rail transport,Rail transport
/m/019jd,Horse,Horse
/m/014fjl,Dog,Dog
/m/011c21,Cat,Cat
/m/01h8_r,Bird,Bird
/m/07swg_w,Domestic animal,Domestic animal
/m/0284p,Animal,Animal
/m/0jbk,Animal sounds,Animal sounds
/m/01h8h_,Livestock,Livestock
/m/07sq7,Frog,Frog
/m/046_k,Insect,Insect
/m/0gj9z,Crab,Crab
/m/0gf0f,Mosquito,Mosquito
/m/07rwf,Fly,Fly
/m/0k_s1,Cricket,Cricket
/m/01p2b,Rooster,Rooster
/m/01p2b_,Crow,Crow
/m/0h__q,Turkey,Turkey
/m/0h05c,Duck,Duck
/m/01x5y,Goose,Goose
/m/02qn99,Chicken,Chicken
/m/01j58n,Fowl,Fowl
/m/0f3_t,Pig,Pig
/m/0j1h1,Cow,Cow
/m/01j5cv,Goat,Goat
/m/01j4z_,Sheep,Sheep
/m/02xqc,Bird vocalization,Bird vocalization
/m/08_zh,Whale vocalization,Whale vocalization
/m/0gypl,Frog vocalization,Frog vocalization
/m/0g09c,Insect vocalization,Insect vocalization
/m/04jc0,Water,Water
/m/0g0x7,Rain,Rain
/m/01_28c,Thunder,Thunder
/m/03cvs,Wind,Wind
/m/0g0j3,Fire,Fire
/m/01j52h,Explosion,Explosion
/m/0h29n,Gunshot,Gunshot
/m/0j47j,Gun,Gun
/m/0gl8z,Weapon,Weapon
/m/01v_0d,Blade,Blade
/m/0gl_x,Boom,Boom
/m/0126g,Fireworks,Fireworks
/m/0gl0j,Crack,Crack
/m/0gk8c,Shatter,Shatter
/m/0gj4s,Rattle,Rattle
/m/0grh8,Pop,Pop
/m/0gjp0,Whizz,Whizz
/m/0gj0p,Whoosh,Whoosh
/m/0gk_b,Splash,Splash
/m/0gk8d,Squish,Squish
/m/0gjl1,Fizz,Fizz
/m/0gjrj,Slosh,Slosh
/m/0gk4r,Gush,Gush
/m/0gk6x,Drip,Drip
/m/0gj7h,Plop,Plop
/m/0j3b3,Raindrop,Raindrop
/m/01sw_y,Stream,Stream
/m/0j1_6,River,River
/m/0gl0k,Waterfall,Waterfall
/m/0j3q9,Ocean,Ocean
/m/0gkq7,Waves,Waves
/m/0h27q,Breaking glass,Breaking glass
/m/0gjsj,Crackle,Crackle
/m/0gj97,Creak,Creak
/m/0gj48,Clatter,Clatter
/m/0gjtg,Squeak,Squeak
/m/0gj4l,Scratch,Scratch
/m/0h4_g,Thump,Thump
/m/0gj4c,Thud,Thud
/m/0gf1m,Rustle,Rustle
/m/0j1h_v,Rumble,Rumble
/m/0gjsn,Click,Click
/m/0gjrj,Clink,Clink
/m/0gk9w,Slap,Slap
/m/0gj0p,Whack,Whack
/m/0gj7j,Stomp,Stomp
/m/0gt21,Banging,Banging
/m/0gkl9,Clap,Clap
/m/0h2k1,Shuffling,Shuffling
/m/0gk5f,Walking,Walking
/m/0kbp_x,Footsteps,Footsteps
/m/0j191,Friction,Friction
/m/01z_9,Scratching,Scratching
/m/01j2d_,Scraping,Scraping
/m/01z_s,Tearing,Tearing
/m/024_d,Squishing,Squishing
/m/01h5x2,Whoosh,Whoosh
/m/0f3_l,Sigh,Sigh
/m/03b_9d,Snore,Snore
/m/014c5w,Cough,Cough
/m/0gjq7,Hiccup,Hiccup
/m/01q4x1,Burp,Burp
/m/06d2h,Fart,Fart
/m/0chx_r,Stutter,Stutter
/m/0h_57,Laughter,Laughter
/m/0j2dx,Snicker,Snicker
/m/0xghr,Gasp,Gasp
/m/0glj9,Giggle,Giggle
/m/014z_1,Chuckle,Chuckle
/m/0hgs8w,Groan,Groan
/m/0j_6v,Panting,Panting
/m/046z3j,Yawn,Yawn
/m/01c48z,Fasp,Fasp
/m/0dzct,Humming,Humming
/m/02zso,Conversation,Conversation
/m/0h499,Speech,Speech
/m/09cpr,Speech,Speech
/m/09x0r,Child speech,Child speech
/m/0dzql,Male speech,Male speech
/m/0kxll,Female speech,Female speech
"""

def load_yamnet_class_names():
    class_names_list = []
    csv_file = io.StringIO(YAMNET_CLASS_MAP_CSV.strip())
    reader = csv.reader(csv_file)
    try:
        next(reader) # Skip header
    except StopIteration:
        print("Waring: YAMNET_CLASS_MAP_CSV seem empty or header only") 
        return np.array([])
    for i, row in enumerate(reader):
        if not row or len(row) < 2: 
            print(f"Warning: Skip {i+1} in YAMNET_CLASS_MAP_CSV because of wrong format: {row}") 
            continue
        class_names_list.append(row[1]) 
    return np.array(class_names_list)

YAMNET_DISPLAY_NAMES = load_yamnet_class_names()

def parse_yamnet_class_map_csv(csv_string):
    f = io.StringIO(csv_string.strip())
    reader = csv.reader(f)
    next(reader) # Skip header
    display_names = []
    for row in reader:
        if len(row) >= 2:
            display_names.append(row[1]) # 'display_name' is the second column
    return display_names

YAMNET_CLASS_NAMES = parse_yamnet_class_map_csv(YAMNET_CLASS_MAP_CSV)

# Thresholds for labelling
TH_VIOLENCE = 0.35
TH_SCREAM = 0.30
TH_BRIGHTNESS_STD = 40.0
TH_TENSION = 0.15