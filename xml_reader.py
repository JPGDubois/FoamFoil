"""
This code reads exported data from XFLR5 in xml format.
A list containing N dictionaries (N is number of sections in the wing) is then produced.
Each dictionary contains all the required information to reproduce the wing geometry.
"""

file = "Main_Wing.xml"

def xml_read(file):
    sectionProperties = {
        "y_position": 0,
        "Chord": 0,
        "Dihedral": 0,
        "Twist": 0,
        "x_number_of_panels": 0,
        "x_panel_distribution": 0,
        "y_number_of_panels": 0,
        "y_panel_distribution": 0,
        "Left_Side_FoilName": 0,
        "Right_Side_FoilName": 0
        }

    f = open(file, "r")
    lines = f.readlines()
    f.close()

    """
    Find the amound of sections that the file consists of.
    Make equal amound of dictionaries as there are sections.
    Read the data from the file and put it in dictionary for the matching section.
    """
    S = []
    inSection = False
    sectionNr = 0
    for line in lines:
        row = line.strip().strip("<")
        value = row[row.find(">") + 1:row.find("</")].strip()
        header = row[:row.find(">")]
        if inSection:
            if header == "/Section":
                inSection = False
                sectionNr += 1
                print("Imported section", sectionNr)
                continue
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
                S[sectionNr][header] = value
                continue
        else:
            if line.strip() == "<Section>":
                inSection = True
                S.append(sectionProperties.copy())
                continue
    return S

print(xml_read(file))
