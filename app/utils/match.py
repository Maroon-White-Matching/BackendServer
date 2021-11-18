from random_word import RandomWords
from nltk.stem.snowball import SnowballStemmer
import Algorithmia
import copy


r = RandomWords()

def apply(input):
    # default weights for the scoring function
    default_weights = {
        "ccr": 1.5,
        "cl": 2.0,
        "cr" : 5.0
    }
    # overwrite the weights if given by user
    if "scoring_weights" in input:
        weights = overwriteWeights(default_weights, input["scoring_weights"])
    else:
        weights = default_weights
    #
    # get the input and do some checking
    #
    validateInput(input)
    # stableMarriageInput = {"optimal": {}, "pessimal": {}}
    x_scoring_list = {}
    y_scoring_list = {}
    # create a preference list for each individual using the scoring function
    for xObject in input["group1"]:
        x_scoring_list[xObject["name"]] = {}
        for yObject in input["group2"]:
            score = scoring_function(weights, xObject, yObject)
            x_scoring_list[xObject["name"]][yObject["name"]] = score
    for yObject in input["group2"]:
        y_scoring_list[yObject["name"]] = {}
        for xObject in input["group1"]:
            score = scoring_function(weights, yObject, xObject)
            y_scoring_list[yObject["name"]][xObject["name"]] = score 
    tmp_x_scoring_list = copy.deepcopy(x_scoring_list)
    tmp_y_scoring_list = copy.deepcopy(y_scoring_list)
    # map & sort the scoring lists into a format that preserves the order of the objects
    for person_x in tmp_x_scoring_list:
        # map into a sortable format
        x_scoring_list[person_x] = list(map(lambda x: {"name": x, "similarity": x_scoring_list[person_x][x]}, x_scoring_list[person_x]))
        # sort the preference list
        x_scoring_list[person_x] = sorted(x_scoring_list[person_x], key=lambda k: k['similarity'], reverse=True)
        # remove the similarity scores from the preference lists
        x_scoring_list[person_x] = list(map(lambda x: x["name"], x_scoring_list[person_x]))
    for person_y in tmp_y_scoring_list:
        # map into a sortable format
        y_scoring_list[person_y] = list(map(lambda x: {"name": x, "similarity": y_scoring_list[person_y][x]}, y_scoring_list[person_y]))
        # sort the preference list
        y_scoring_list[person_y] = sorted(y_scoring_list[person_y], key=lambda k: k['similarity'], reverse=True)
        # remove the similarity scores from the preference lists
        y_scoring_list[person_y] = list(map(lambda x: x["name"], y_scoring_list[person_y]))
    # if one group has a larger preference list, add null characters to the end of the list
    # this is to ensure that the stable marriage algorithm works properly
    group_difference = len(x_scoring_list) - len(y_scoring_list)
    null_people = []
    if group_difference == 0.0:
        # create stable pairs using the given preference lists with the stable marriage algorithm
        stable_marriage_input = {
            "optimal": x_scoring_list,
            "pessimal": y_scoring_list
        }
    else:
        if group_difference > 0:
            for i in range(group_difference):
                null_y = randomword(20)
                null_people.append(null_y)
                y_scoring_list[null_y] = []
                for person_x in x_scoring_list:
                    x_scoring_list[person_x].append(null_y)
                    y_scoring_list[null_y].append(person_x)
        elif group_difference < 0:
            for i in range(group_difference):
                null_x = randomword(20)
                null_people.append(null_x)
                x_scoring_list[null_y] = []
                for person_y in y_scoring_list:
                    y_scoring_list[person_y].append(null_x)
                    x_scoring_list[null_y].append(person_y)
                    
        # create stable pairs using the given preference lists with the stable marriage algorithm
        stable_marriage_input = {
            "optimal": x_scoring_list,
            "pessimal": y_scoring_list
        }
        
    stable_marriages = Algorithmia.client('simxKKjq5TG0tcJsqUOycPt6KgP1').algo("matching/StableMarriageAlgorithm").pipe(stable_marriage_input).result["matches"]

    if group_difference == 0.0:
        return stable_marriages
    elif group_difference > 0:
        tmp = copy.deepcopy(stable_marriages)
        stable_marriages = dict((v,k) for k,v in tmp.items())
        for person in null_people:
            stable_marriages.pop(person)
        return stable_marriages
    elif group_difference < 0:
        for person in null_people:
            stable_marriages.pop(person)
        return stable_marriages

def randomword(length):
    return ''.join(r.get_random_word() for i in range(length))

def overwriteWeights(default, new):
    rVal = default
    if "ccr" in new:
        rVal["ccr"] = float(new["ccr"])
    if "cl" in new:
        rVal["cl"] = float(new["cl"])
    if "cr" in new:
        rVal["cr"] = float(new["cr"])
    return rVal
    
def scoring_function(weights, person1, person2):
    # returns a score that gives the similarity between 2 people
    # scoring function:
    #   +add for each interest * weight
    #   +add for each value * weight
    #   -subtract meeting difference * weight
    #   -subtract location difference * weight
    ss = SnowballStemmer("english")
    score = 0.0
    major_list1 = person1["ccr"]
    major_list2 = person2["ccr"]
    # compare similar major
    for major1 in major_list1:
        for major2 in major_list2:
            stem1 = ss.stem(major1.lower())
            stem2 = ss.stem(major2.lower())
            
            if stem1 == stem2:
                score += weights["ccr"]
    if "cr" in person1 and "cr" in person2:
        personality_list1 = person1["cr"]
        personality_list2 = person2["cr"]  
        for value1 in personality_list1:
            for value2 in personality_list2:
                stem1 = ss.stem(value1.lower())
                stem2 = ss.stem(value2.lower())
            if stem1 == stem2:
                score += weights["cr"]    
    # score proximity of the paired couple if distance exists for each person
    if "cl" in person1 and "cl" in person2:
        d1 = float(person1["cl"])
        d2 = float(person2["cl"])
        score -= abs(d1 - d2) * weights["cl"] # student 1 = 0.4810929929 , person 2 = 0.561982938  
    return score    
    
def validateInput(input):
    # Validate the initial input fields
    if "group1" not in input and "group2" not in input:
        raise Exception("Please provide both the person_x and person_y groups")
    elif "group2" not in input:
        raise Exception("Please provide the person_y group.")
    elif "group1" not in input:
        raise Exception("Please provide the person_x group.")
    # The only required field for a user object is "name"
    for x in ["group1", "group2"]:
        if not isinstance(input[x], list):
            raise Exception("Please provide a list of people for each group.")      
        if len(input[x]) == 0:
            raise Exception("Groups cannot be empty.")
        for person in input[x]:
            if "name" not in person:
                raise Exception("Please provide the name all people.")    
            # if not isinstance(person["major"], list):
            #     raise Exception("Please provide a list of major for each person.")             
            # Check validity for the longitude and latitude if the distance field exists
            if "distance" in person:
                if not isinstance(person["distance"], dict):
                    raise Exception("Please provide valid distance")
                if "lat" not in person["distance"] or "long" not in person["distance"]:
                    raise Exception("Please provide valid distance")
                if not isinstance(person["distance"]["lat"], float) or not isinstance(person["distance"]["long"], float):
                    raise Exception("coordinate interests can only be in float.")
    # unequal groups are now supported
    # if len(input["group1"]) != len(input["group2"]):
    #     raise Exception("The size of both groups should be same.")
    