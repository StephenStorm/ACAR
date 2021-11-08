
id_class_dict = [
        {'name': 'bend/bow (at the waist)', 'id': 1}, {'name': 'crouch/kneel', 'id': 3}, {'name': 'dance', 'id': 4}, {'name': 'fall down', 'id': 5}, 
        {'name': 'get up', 'id': 6}, {'name': 'jump/leap', 'id': 7}, {'name': 'lie/sleep', 'id': 8}, {'name': 'martial art', 'id': 9}, 
        {'name': 'run/jog', 'id': 10}, {'name': 'sit', 'id': 11}, {'name': 'stand', 'id': 12}, {'name': 'swim', 'id': 13}, 
        {'name': 'walk', 'id': 14}, {'name': 'answer phone', 'id': 15}, {'name': 'carry/hold (an object)', 'id': 17}, 
        {'name': 'climb (e.g., a mountain)', 'id': 20}, {'name': 'close (e.g., a door, a box)', 'id': 22}, {'name': 'cut', 'id': 24}, 
        {'name': 'dress/put on clothing', 'id': 26}, {'name': 'drink', 'id': 27}, {'name': 'drive (e.g., a car, a truck)', 'id': 28}, 
        {'name': 'eat', 'id': 29}, {'name': 'enter', 'id': 30}, {'name': 'hit (an object)', 'id': 34}, {'name': 'lift/pick up', 'id': 36}, 
        {'name': 'listen (e.g., to music)', 'id': 37}, {'name': 'open (e.g., a window, a car door)', 'id': 38}, {'name': 'play musical instrument', 'id': 41}, 
        {'name': 'point to (an object)', 'id': 43}, {'name': 'pull (an object)', 'id': 45}, {'name': 'push (an object)', 'id': 46}, {'name': 'put down', 'id': 47}, 
        {'name': 'read', 'id': 48}, {'name': 'ride (e.g., a bike, a car, a horse)', 'id': 49}, {'name': 'sail boat', 'id': 51}, {'name': 'shoot', 'id': 52}, 
        {'name': 'smoke', 'id': 54}, {'name': 'take a photo', 'id': 56}, {'name': 'text on/look at a cellphone', 'id': 57}, {'name': 'throw', 'id': 58}, 
        {'name': 'touch (an object)', 'id': 59}, {'name': 'turn (e.g., a screwdriver)', 'id': 60}, {'name': 'watch (e.g., TV)', 'id': 61}, 
        {'name': 'work on a computer', 'id': 62}, {'name': 'write', 'id': 63}, {'name': 'fight/hit (a person)', 'id': 64}, 
        {'name': 'give/serve (an object) to (a person)', 'id': 65}, {'name': 'grab (a person)', 'id': 66}, {'name': 'hand clap', 'id': 67}, 
        {'name': 'hand shake', 'id': 68}, {'name': 'hand wave', 'id': 69}, {'name': 'hug (a person)', 'id': 70}, {'name': 'kiss (a person)', 'id': 72}, 
        {'name': 'lift (a person)', 'id': 73}, {'name': 'listen to (a person)', 'id': 74}, {'name': 'push (another person)', 'id': 76}, 
        {'name': 'sing to (e.g., self, a person, a group)', 'id': 77}, {'name': 'take (an object) from (a person)', 'id': 78}, 
        {'name': 'talk to (e.g., self, a person, a group)', 'id': 79}, {'name': 'watch (a person)', 'id': 80}
]


id2class = {
    1: 'bend/bow (at the waist)', 3: 'crouch/kneel', 4: 'dance', 5: 'fall down', 6: 'get up', 7: 'jump/leap', 8: 'lie/sleep', 9: 'martial art', 10: 'run/jog', 
    11: 'sit', 12: 'stand', 13: 'swim', 14: 'walk', 15: 'answer phone', 17: 'carry/hold (an object)', 20: 'climb (e.g., a mountain)', 
    22: 'close (e.g., a door, a box)', 24: 'cut', 26: 'dress/put on clothing', 27: 'drink', 28: 'drive (e.g., a car, a truck)', 29: 'eat', 30: 'enter', 
    34: 'hit (an object)', 36: 'lift/pick up', 37: 'listen (e.g., to music)', 38: 'open (e.g., a window, a car door)', 41: 'play musical instrument', 
    43: 'point to (an object)', 45: 'pull (an object)', 46: 'push (an object)', 47: 'put down', 48: 'read', 49: 'ride (e.g., a bike, a car, a horse)', 
    51: 'sail boat', 52: 'shoot', 54: 'smoke', 56: 'take a photo', 57: 'text on/look at a cellphone', 58: 'throw', 59: 'touch (an object)', 
    60: 'turn (e.g., a screwdriver)', 61: 'watch (e.g., TV)', 62: 'work on a computer', 63: 'write', 64: 'fight/hit (a person)', 
    65: 'give/serve (an object) to (a person)', 66: 'grab (a person)', 67: 'hand clap', 68: 'hand shake', 69: 'hand wave', 70: 'hug (a person)', 
    72: 'kiss (a person)', 73: 'lift (a person)', 74: 'listen to (a person)', 76: 'push (another person)', 77: 'sing to (e.g., self, a person, a group)', 
    78: 'take (an object) from (a person)', 79: 'talk to (e.g., self, a person, a group)', 80: 'watch (a person)'
}

class2id = {
    'bend/bow (at the waist)': 1, 'crouch/kneel': 3, 'dance': 4, 'fall down': 5, 'get up': 6, 'jump/leap': 7, 'lie/sleep': 8, 'martial art': 9, 
    'run/jog': 10, 'sit': 11, 'stand': 12, 'swim': 13, 'walk': 14, 'answer phone': 15, 'carry/hold (an object)': 17, 'climb (e.g., a mountain)': 20, 
    'close (e.g., a door, a box)': 22, 'cut': 24, 'dress/put on clothing': 26, 'drink': 27, 'drive (e.g., a car, a truck)': 28, 'eat': 29, 'enter': 30, 
    'hit (an object)': 34, 'lift/pick up': 36, 'listen (e.g., to music)': 37, 'open (e.g., a window, a car door)': 38, 'play musical instrument': 41, 
    'point to (an object)': 43, 'pull (an object)': 45, 'push (an object)': 46, 'put down': 47, 'read': 48, 'ride (e.g., a bike, a car, a horse)': 49, 
    'sail boat': 51, 'shoot': 52, 'smoke': 54, 'take a photo': 56, 'text on/look at a cellphone': 57, 'throw': 58, 'touch (an object)': 59, 
    'turn (e.g., a screwdriver)': 60, 'watch (e.g., TV)': 61, 'work on a computer': 62, 'write': 63, 'fight/hit (a person)': 64, 
    'give/serve (an object) to (a person)': 65, 'grab (a person)': 66, 'hand clap': 67, 'hand shake': 68, 'hand wave': 69, 'hug (a person)': 70, 
    'kiss (a person)': 72, 'lift (a person)': 73, 'listen to (a person)': 74, 'push (another person)': 76, 'sing to (e.g., self, a person, a group)': 77, 
    'take (an object) from (a person)': 78, 'talk to (e.g., self, a person, a group)': 79, 'watch (a person)': 80
}

'''

class2id = dict()
id2class = dict()
for name_id in id_class_dict :
    name = name_id['name']
    id = name_id['id']
    class2id[name] = id
    id2class[id] = name
print(id2class)

print(class2id)
'''