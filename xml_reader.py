file = "Main_Wing.xml"

f = open(file, "r")
lines = f.readlines()
f.close()

sectionAmount = 0

for line in lines:
    sectionAmount += line.count("<Section>")
print(sectionAmount)
class Section:
    def __init__(self, sectionNr):
        self.sectionNr = sectionNr
        self.dict = {}










Section1 = Section(4)
print(Section1.dict)
