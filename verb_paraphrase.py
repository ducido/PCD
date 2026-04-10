# import simpler_env
# import pandas as pd
# from tqdm import tqdm

# tasks = (
#     "google_robot_pick_coke_can",
#     "google_robot_move_near",
#     "google_robot_close_drawer",
#     "google_robot_open_drawer",
#     "google_robot_place_apple_in_closed_top_drawer",
#     "widowx_carrot_on_plate",
#     "widowx_spoon_on_towel",
#     "widowx_put_eggplant_in_basket",
#     "widowx_stack_cube",
# )

# seen = set()
# data = []

# for task in tqdm(tasks, desc="Tasks"):
#     env = simpler_env.make(task)
    
#     for i in tqdm(range(100), desc=f"{task}", leave=False):
#         obs, _ = env.reset(seed=i)
        
#         instruction = env.unwrapped.get_language_instruction()
#         is_final_subtask = env.unwrapped.is_final_subtask()
        
#         key = (task, instruction, is_final_subtask)

#         if key not in seen:
#             seen.add(key)
#             data.append({
#                 "task": task,
#                 "instruction": instruction,
#                 "is_final_subtask": is_final_subtask
#             })

#     del env

# # tạo dataframe
# df = pd.DataFrame(data)

# # save file
# df.to_csv("robot_tasks.csv", index=False)

# print(df.head())



import pandas as pd
import random

# =====================
# 1. Controlled vocab
# =====================
VERB_MAP = {
    "move": ["move", "place", "put", "bring"],
    "pick": ["pick", "grab", "take"],
    "close": ["close", "shut"],
    "open": ["open"],
    "put": ["put", "place"],
    "stack": ["stack", "place"]
}

REL_MAP = {
    "near": ["near", "next to", "close to"],
    "on": ["on", "on top of"],
    "into": ["into", "inside"]
}

# =====================
# 2. Paraphrase function
# =====================
def structured_paraphrase(instr):
    words = instr.lower().split()

    # -------- move / near --------
    if "near" in words:
        idx = words.index("near")
        verb = words[0]
        obj1 = " ".join(words[1:idx])
        obj2 = " ".join(words[idx+1:])

        verb_new = random.choice(VERB_MAP.get(verb, [verb]))
        rel_new = random.choice(REL_MAP["near"])

        return f"{verb_new} {obj1} {rel_new} {obj2}"

    # -------- on --------
    if "on" in words:
        idx = words.index("on")
        verb = words[0]
        obj1 = " ".join(words[1:idx])
        obj2 = " ".join(words[idx+1:])

        verb_new = random.choice(VERB_MAP.get(verb, [verb]))
        rel_new = random.choice(REL_MAP["on"])

        return f"{verb_new} {obj1} {rel_new} {obj2}"

    # -------- into --------
    if "into" in words:
        idx = words.index("into")
        verb = words[0]
        obj1 = " ".join(words[1:idx])
        obj2 = " ".join(words[idx+1:])

        verb_new = random.choice(VERB_MAP.get(verb, [verb]))
        rel_new = random.choice(REL_MAP["into"])

        return f"{verb_new} {obj1} {rel_new} {obj2}"

    # -------- simple commands --------
    if len(words) >= 2:
        verb = words[0]
        rest = " ".join(words[1:])
        verb_new = random.choice(VERB_MAP.get(verb, [verb]))
        return f"{verb_new} {rest}"

    return instr


# =====================
# 3. Apply to dataframe
# =====================
df = pd.read_csv("robot_tasks.csv")

paraphrased = []
for instr in df["instruction"]:
    para = structured_paraphrase(instr)

    # tránh trùng hoàn toàn → regenerate nhẹ
    if para == instr:
        para = structured_paraphrase(instr)

    paraphrased.append(para)

df["paraphrased_instruction"] = paraphrased

# =====================
# 4. Save
# =====================
df.to_csv("robot_tasks_with_paraphrase.csv", index=False)

print(df.head())