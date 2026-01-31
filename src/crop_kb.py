"""
Knowledge Base for Smart Agri XAI.
Contains detailed cultivation guides, disease management, and protection protocols.
Includes Ideal NPk/Weather params for visualization.
Added: Market Prices for Economic Analysis.
"""

CROP_KNOWLEDGE_BASE = {
    'Rice': {
        'cultivation': [
            "**1. Seed Selection & Treatment**: Choose high-yielding semi-dwarf varieties (like IR-64, Swarna) resistant to local pests. Treat seeds with Carbendazim (2g/kg) to prevent seed-borne diseases.",
            "**2. Nursery Preparation**: Raised wet-bed nurseries are ideal. Calculate 1/10th of the main field area for the nursery. Sow sprouted seeds and maintain shallow water level for 20-25 days.",
            "**3. Main Field Preparation**: Puddle the field 3-4 times with standing water to destroy weeds and reduce percolation. Level the field perfectly using a plank.",
            "**4. Transplanting**: Transplant 2-3 seedlings per hill at a spacing of 20x10 cm. Ensure seedlings are upright and not buried too deep.",
            "**5. Water Management**: Maintain 2-5 cm of water throughout the vegetative phase. Drain the field 10 days before harvest to hasten muturity.",
            "**6. Nutrient Management**: Apply N-P-K in split doses. Apply 50% Nitrogen at transplanting, 25% at tillering, and 25% at panicle initiation.",
            "**7. Harvesting**: Harvest when 80% of the panicles turn golden yellow. Thresh immediately and dry grains to 14% moisture."
        ],
        'diseases': [
            "**Blast**: Small spindle-shaped spots on leaves with grey centers. *Remedy*: Spray Tricyclazole 75 WP.",
            "**Bacterial Leaf Blight**: Water-soaked streaks on leaf edges turning yellow. *Remedy*: Avoid excessive Nitrogen; Spray Copper Oxychloride.",
            "**Stem Borer**: Causes 'Dead Heart' in young plants and 'White Ear' in older ones. *Remedy*: Apply Cartap Hydrochloride granules."
        ],
        'protection': "Ensure proper drainage channels to prevent waterlogging during heavy rains. In cyclone-prone areas, use lodging-resistant varieties. Monitor for Brown Plant Hopper (BPH) at the base of the plant.",
        'ideal': {'N': 80, 'P': 40, 'K': 40, 'temp': 25, 'hum': 80, 'ph': 6.5, 'rain': 200},
        'price': 22000 # INR per Ton
    },
    'Maize': {
        'cultivation': [
             "**1. Land Preparation**: Maize requires a fine tilth. Plough the land 2-3 times and break clods. Form ridges and furrows to ensure excellent drainage, as Maize is highly sensitive to waterlogging.",
             "**2. Sowing**: Sow seeds on the side of ridges at a spacing of 60x20 cm. Best sowing time is onset of monsoon (Kharif) or Oct-Nov (Rabi). Seed rate: 20 kg/ha.",
             "**3. Weed Control**: Apply Atrazine as a pre-emergence herbicide within 48 hours of sowing. Hand weeding at 20-25 days is beneficial.",
             "**4. Fertilizer Application**: Maize is a heavy feeder. Apply N-P-K (120:60:40). Nitrogen should be applied in 3 splits: Basal, Knee-high stage, and Tasseling stage.",
             "**5. Irrigation**: Critical stages are Knee-high, Tasseling, and Silking. Water stress at these stages significantly reduces yield.",
             "**6. Harvesting**: Harvest when the cob sheath turns pale yellow and dry. Grains should be hard and dry."
        ],
        'diseases': [
            "**Turcicum Leaf Blight**: Long elliptical grey/brown spots. *Remedy*: Spray Mancozeb or Zineb.",
            "**Fall Armyworm (FAW)**: Larvae feed on whorls causing ragging. *Remedy*: Install pheromone traps; Apply Emamectin Benzoate or Spinetoram."
        ],
        'protection': "Ensure drainage is effective; stagnant water for even 24 hours can kill the crop. Protect from parrots/birds during cob maturation.",
        'ideal': {'N': 100, 'P': 60, 'K': 40, 'temp': 24, 'hum': 60, 'ph': 6.5, 'rain': 100},
        'price': 20000 
    },
    'Cotton': {
        'cultivation': [
            "**1. Soil & Climate**: Deep black cotton soils (regur) are best. Requires a frost-free season of 180-200 days.",
            "**2. Land Preparation**: Deep ploughing once every 3 years. Harrowing 2-3 times to create a fine seedbed. Form ridges and furrows.",
            "**3. Sowing**: Sow hybrid/Bt seeds at spacing 90x60 cm or 120x60 cm depending on variety. Dibble 2 seeds per hill. Seed treatment with Imidacloprid prevents sucking pests.",
            "**4. Nutrient Management**: Application of FYM is crucial. N-P-K (120:60:60). Apply Nitrogen in 3 splits (Basal, Square formation, Flowering). Spray Magnesium Sulphate if leaves turn red (reddening).",
            "**5. Weed Control**: Keep field weed-free for the first 60 days. Inter-cultivation/Earthing up helps in weed control and soil aeration.",
            "**6. Topping**: Nip the terminal bud after 90 days to encourage side branching and boll formation.",
            "**7. Harvesting**: Pick fully burst bolls during dry hours (morning). Avoid mixing dried leaves with cotton."
        ],
        'diseases': [
            "**Pink Bollworm**: Larvae bore into bolls, staining lint. *Remedy*: Use Pheromone traps; Spray Profenofos or Indoxacarb.",
            "**Cotton Leaf Curl Virus (CLCuV)**: Upward curling of leaves, transmitted by Whitefly. *Remedy*: Control vector using Diafenthiuron or Acetamiprid.",
            "**Sucking Pests (Jassids/Thrips)**: Yellowing/curling. *Remedy*: Systemic insecticides."
        ],
        'protection': "Heavy rain during boll bursting ruins quality; harvest immediately if rain is forecast. Avoid water stress during flowering to prevent boll shedding.",
        'ideal': {'N': 120, 'P': 40, 'K': 20, 'temp': 27, 'hum': 50, 'ph': 7.0, 'rain': 80},
        'price': 60000
    },
    'Chickpea': {
        'cultivation': [
            "**1. Land Preparation**: Requires a rough seedbed for better aeration. One deep ploughing followed by harrowing.",
            "**2. Sowing**: Ideal time is Oct-Nov. Deep sowing (10cm) allows roots to access residual soil moisture and prevents wilt.",
            "**3. Spacing & Seed Rate**: 30x10 cm. Seed rate 75-100 kg/ha depending on seed size.",
            "**4. Nipping**: Pluck the apical buds when plant is 15-20 cm height to encourage branching and more pods.",
            "**5. Irrigation**: Generally rainfed. If available, one irrigation at branching and one at pod filling is sufficient. Avoid irrigating at flowering.",
            "**6. Harvesting**: Harvest when leaves turn reddish-brown and start shedding. Plants are pulled out or cut."
        ],
        'diseases': [
            "**Fusarium Wilt**: Sudden drying/drooping of plants. *Remedy*: Seed treatment with Trichoderma viride; Follow crop rotation (3 years).",
            "**Gram Pod Borer (Helicoverpa)**: Larva feeds on leaves and pods. *Remedy*: Install bird perches; Spray HaNPV or Indoxacarb."
        ],
        'protection': "Chickpea is highly sensitive to frost. Light irrigation or smoking the field can protect from frost damage.",
        'ideal': {'N': 40, 'P': 60, 'K': 80, 'temp': 20, 'hum': 40, 'ph': 7.0, 'rain': 80},
        'price': 52000
    },
    'Groundnut': {
        'cultivation': [
            "**1. Soil**: Well-drained sandy loam is best to allow easy peg penetration. Avoid clay soils.",
            "**2. Seed Treatment**: Treating seeds with Rhizobium culture increases nodulation and Nitrogen fixation.",
            "**3. Sowing**: Kharif (June-July) or Rabi (Nov-Dec). Spacing 30x10 cm. Use broad-bed and furrow method for better yield.",
            "**4. Gypsum Application**: Apply Gypsum @ 500 kg/ha at flowering (40-45 DAS). This provides Calcium for pod filling and Sulphur for oil content.",
            "**5. Water**: Critical stages are Flowering, Pegging, and Pod formation. Avoid moisture stress at pod development.",
            "**6. Harvesting**: Harvest when the inside of the shell turns dark brown. Pull out plants and strip pods."
        ],
        'diseases': [
            "**Tikka Disease (Leaf Spot)**: Dark spots with yellow halos. *Remedy*: Spray Carbendazim + Mancozeb.",
            "**Rust**: Orange pustules on leaves. *Remedy*: Spray Chlorothalonil.",
            "**Stem Rot**: Rotting at collar region. *Remedy*: Soil drenching with Trichoderma."
        ],
        'protection': "Prevent Aflatoxin contamination by drying pods thoroughly (<9% moisture) and avoiding damage to shells during harvest.",
        'ideal': {'N': 20, 'P': 60, 'K': 40, 'temp': 28, 'hum': 50, 'ph': 6.5, 'rain': 70},
        'price': 58000
    },
    'Tea': {
        'cultivation': [
            "**1. Climate & Soil**: Requires cool, humid climate and acidic soil (pH 4.5-5.5). Altitude 1000-2000m is ideal.",
            "**2. Propagation**: Vegetative propagation using single node cuttings in nursery beds.",
            "**3. Planting**: Plant in pits or trenches. Spacing 105x65 cm. Mulching is essential to conserve moisture and suppress weeds.",
            "**4. Pruning**: Essential to maintain the bush in a vegetative phase and keep the plucking table manageable. Types: Centering, Tipping, Skiffing.",
            "**5. Plucking**: Harvest 'Two leaves and a bud'. Determine plucking round based on flush growth (usually 7-10 days).",
            "**6. Manuring**: Regular application of NPK mixtures (ratio 2:1:2 or 2:1:3) is needed for continuous vegetative growth."
        ],
        'diseases': [
            "**Blister Blight**: Blisters on young leaves. *Remedy*: Spray Copper Oxychloride + Hexaconazole.",
            "**Red Spider Mite**: Leaves turn reddish-bronze. *Remedy*: Spray Dicofol or Propargite.",
            "**Mosquito Bug**: Punctures leaves. *Remedy*: Spray Thiamethoxam."
        ],
        'protection': "Provide shade using Silver Oak or Albizia trees (regulated shade). Protect from frost in high altitudes.",
        'ideal': {'N': 100, 'P': 40, 'K': 80, 'temp': 22, 'hum': 90, 'ph': 5.0, 'rain': 250},
        'price': 150000 
    },
    'Coffee': {
        'cultivation': ["**1. Shade**: Establish shade trees (Silver Oak) 1-2 years prior.", "**2. Sowing**: Seeds or Clonal cuttings.", "**3. Training**: Single stem system. Topping at 5ft.", "**4. Berry picking**: Fly picking -> Main picking -> Stripping."],
        'diseases': ["**Coffee Rust**: Orange dust. *Remedy*: Bordeaux mixture.", "**Berry Borer**: Beetle inside berry."],
        'protection': "Protect from direct afternoon sun and frost.",
        'ideal': {'N': 100, 'P': 60, 'K': 80, 'temp': 23, 'hum': 60, 'ph': 6.0, 'rain': 150},
        'price': 180000
    }
}
