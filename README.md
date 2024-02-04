Data

Predicting forest cover type from cartographic variables only
(no remotely sensed data). The actual forest cover type for
a given observation (30 x 30 meter cell) was determined from
US Forest Service (USFS) Region 2 Resource Information System
(RIS) data. Independent variables were derived from data
originally obtained from US Geological Survey (USGS) and
USFS data. Data is in raw form (not scaled) and contains
binary (0 or 1) columns of data for qualitative independent
variables (wilderness areas and soil types).

This study area includes four wilderness areas located in the
Roosevelt National Forest of northern Colorado. These areas
represent forests with minimal human-caused disturbances,
so that existing forest cover types are more a result of
ecological processes rather than forest management practices.

Some background information for these four wilderness areas:
Neota (area 2) probably has the highest mean elevational value of
the 4 wilderness areas. Rawah (area 1) and Comanche Peak (area 3)
would have a lower mean elevational value, while Cache la Poudre
(area 4) would have the lowest mean elevational value.

As for primary major tree species in these areas, Neota would have
spruce/fir (type 1), while Rawah and Comanche Peak would probably
have lodgepole pine (type 2) as their primary species, followed by
spruce/fir and aspen (type 5). Cache la Poudre would tend to have
Ponderosa pine (type 3), Douglas-fir (type 6), and
cottonwood/willow (type 4).

The Rawah and Comanche Peak areas would tend to be more typical of
the overall dataset than either the Neota or Cache la Poudre, due
to their assortment of tree species and range of predictive
variable values (elevation, etc.) Cache la Poudre would probably
be more unique than the others, due to its relatively low
elevation range and species composition.

Attribute information:

Given is the attribute name, attribute type, the measurement unit and
a brief description. The forest cover type is the classification
problem. The order of this listing corresponds to the order of
numerals along the rows of the database.

    Elevation, quantitative (meters): Elevation in meters
    Aspect, quantitative (azimuth): Aspect in degrees azimuth
    Slope, quantitative (degrees): Slope in degrees
    Horizontal_Distance_To_Hydrology , quantitative (meters): Horz Dist to nearest surface water features
    Vertical_Distance_To_Hydrology , quantitative (meters): Vert Dist to nearest surface water features
    Horizontal_Distance_To_Roadways , quantitative (meters ): Horz Dist to nearest roadway
    Hillshade_9am , quantitative (0 to 255 index): Hillshade index at 9am, summer solstice
    Hillshade_Noon, quantitative (0 to 255 index): Hillshade index at noon, summer soltice
    Hillshade_3pm, quantitative (0 to 255 index): Hillshade index at 3pm, summer solstice
    Horizontal_Distance_To_Fire_Points, quantitative (meters): Horz Dist to nearest wildfire ignition points
    Wilderness_Area (4 binary columns), qualitative (0 (absence) or 1 (presence)): Wilderness area designation
    Soil_Type (40 binary columns), qualitative ( 0 (absence) or 1 (presence)): Soil Type designation
    Cover_Type (7 types), integer (1 to 7): Forest Cover Type designation

Code Designations:

Wilderness Areas:

    1 -- Rawah Wilderness Area
    2 -- Neota Wilderness Area
    3 -- Comanche Peak Wilderness Area
    4 -- Cache la Poudre Wilderness Area

Soil Types: 1 to 40 : based on the USFS Ecological Landtype Units (ELUs) for this study area:

    1: ELU 2702, Cathedral family - Rock outcrop complex, extremely stony.
    2: ELU 2703, Vanet - Ratake families complex, very stony.
    3: ELU 2704, Haploborolis - Rock outcrop complex, rubbly.
    4: ELU 2705, Ratake family - Rock outcrop complex, rubbly.
    5: ELU 2706, Vanet family - Rock outcrop complex complex, rubbly.
    6: ELU 2717, Vanet - Wetmore families - Rock outcrop complex, stony.
    7: ELU 3501, Gothic family.
    8: ELU 3502, Supervisor - Limber families complex.
    9: ELU 4201, Troutville family, very stony.
    10: ELU 4703, Bullwark - Catamount families - Rock outcrop complex, rubbly.
    11: ELU 4704, Bullwark - Catamount families - Rock land complex, rubbly.
    12: ELU 4744, Legault family - Rock land complex, stony.
    13: ELU 4758, Catamount family - Rock land - Bullwark family complex, rubbly.
    14: ELU 5101, Pachic Argiborolis - Aquolis complex.
    15: ELU 5151, unspecified in the USFS Soil and ELU Survey.
    16: ELU 6101, Cryaquolis - Cryoborolis complex.
    17: ELU 6102, Gateview family - Cryaquolis complex.
    18: ELU 6731, Rogert family, very stony.
    19: ELU 7101, Typic Cryaquolis - Borohemists complex.
    20: ELU 7102, Typic Cryaquepts - Typic Cryaquolls complex.
    21: ELU 7103, Typic Cryaquolls - Leighcan family, till substratum complex.
    22: ELU 7201, Leighcan family, till substratum, extremely bouldery.
    23: ELU 7202, Leighcan family, till substratum - Typic Cryaquolls complex.
    24: ELU 7700, Leighcan family, extremely stony.
    25: ELU 7701, Leighcan family, warm, extremely stony.
    26: ELU 7702, Granile - Catamount families complex, very stony.
    27: ELU 7709, Leighcan family, warm - Rock outcrop complex, extremely stony.
    28: ELU 7710, Leighcan family - Rock outcrop complex, extremely stony.
    29: ELU 7745, Como - Legault families complex, extremely stony.
    30: ELU 7746, Como family - Rock land - Legault family complex, extremely stony.
    31: ELU 7755, Leighcan - Catamount families complex, extremely stony.
    32: ELU 7756, Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
    33: ELU 7757, Leighcan - Catamount families - Rock outcrop complex, extremely stony.
    34: ELU 7790, Cryorthents - Rock land complex, extremely stony.
    35: ELU 8703, Cryumbrepts - Rock outcrop - Cryaquepts complex.
    36: ELU 8707, Bross family - Rock land - Cryumbrepts complex, extremely stony.
    37: ELU 8708, Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
    38: ELU 8771, Leighcan - Moran families - Cryaquolls complex, extremely stony.
    39: ELU 8772, Moran family - Cryorthents - Leighcan family complex, extremely stony.
    40: ELU 8776, Moran family - Cryorthents - Rock land complex, extremely stony.

Note: the ELU is comprised of four digits

First digit: climatic zone

    1: lower montane dry
    2: lower montane
    3: montane dry
    4: montane
    5: montane dry and montane
    6: montane and subalpine
    7: subalpine
    8: alpine

Second digit: geologic zones

    1: alluvium
    2: glacial
    3: shale
    4: sandstone
    5: mixed sedimentary
    6: unspecified in the USFS ELU Survey
    7: igneous and metamorphic
    8: volcanic

The third and fourth ELU digits are unique to the mapping unit
and have no special meaning to the climatic or geologic zones.

Forest Cover Type Classes:

    1 - Spruce/Fir
    2 - Lodgepole Pine
    3 - Ponderosa Pine
    4 - Cottonwood/Willow
    5 - Aspen
    6 - Douglas-fir
    7 - Krummholz
