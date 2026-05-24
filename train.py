import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Create a highly detailed synthetic dataset
np.random.seed(42)
n_samples = 1500

ages = np.random.randint(18, 65, n_samples)
bmis = np.random.uniform(15.0, 40.0, n_samples)
goals = np.random.randint(0, 3, n_samples)

# Detailed plan text blocks
weight_loss_plan = (
    "🏠 7-DAY HOME WORKOUT PLAN (No Equipment):\n"
    "• Mon: Full Body HIIT Burn (4 Rounds: 45s work, 15s rest) - Jumping Jacks, Squats, Push-ups, Mountain Climbers, Plank.\n"
    "• Tue: Core & Cardio - 30 min brisk walk/jog + 3 sets of 15 Crunches, 20 Bicycle Crunches, 45s Flutter Kicks.\n"
    "• Wed: Lower Body Intervals (4 Rounds: 40s work, 20s rest) - Lunges, Sumo Squats, Glute Bridges, Skater Hops.\n"
    "• Thu: Active Recovery - 20 mins of full-body dynamic stretching or yoga.\n"
    "• Fri: Upper Body Focus (4 Rounds: 45s work, 15s rest) - Pike Push-ups, Standard Push-ups, Chair Dips, Superman Holds.\n"
    "• Sat: Weekend Fat Burner (5 Rounds fast) - 15 Burpees, 20 Jump Squats, 15 Diamond Push-ups, 1 min Plank.\n"
    "• Sun: Complete Rest & Hydration Day.\n\n"
    "🥗 MON-TO-SUN CALORIC DEFICIT DIET PLAN:\n"
    "• Mon-Wed:\n"
    "  - Breakfast: 3 Scrambled Egg Whites + 1 Whole Egg + ½ cup Oatmeal.\n"
    "  - Lunch: 150g Grilled Chicken Breast/Tofu + Large Green Salad with lemon dressing.\n"
    "  - Snack: 1 cup low-fat Greek yogurt or 1 scoop protein shake + 10 almonds.\n"
    "  - Dinner: 150g Grilled Fish/Paneer + 1 cup steamed broccoli + ½ cup brown rice.\n"
    "• Thu (Low Carb):\n"
    "  - Breakfast: Protein smoothie (1 scoop protein, 1 cup spinach, ½ banana, almond milk).\n"
    "  - Lunch: 1 cup boiled Chickpea salad with onions, tomatoes, cucumber, and lime.\n"
    "  - Snack: 1 sliced cucumber + 2 tbsp hummus.\n"
    "  - Dinner: Massive vegetable stir-fry + 150g shredded chicken breast or paneer.\n"
    "• Fri-Sat:\n"
    "  - Breakfast: 2 slices whole-wheat toast + ½ mashed avocado + 2 boiled eggs.\n"
    "  - Lunch: Lentil Soup (Daal) + 150g grilled chicken breast/soy chunks + raw veggies.\n"
    "  - Snack: 1 medium apple + 1 cup green tea.\n"
    "  - Dinner: 150g Baked Fish/Tofu + stir-fried bell peppers/mushrooms + ½ cup quinoa.\n"
    "• Sun:\n"
    "  - Breakfast: Oatmeal pancake (½ cup oats blended with 3 egg whites and ½ banana).\n"
    "  - Lunch: Leftover chicken/paneer/tofu stir-fry paired with a large side salad.\n"
    "  - Snack: Handful of roasted makhana (fox nuts) or walnuts.\n"
    "  - Dinner: Light clear vegetable soup with 100g chunks of lean chicken or tofu."
)

muscle_gain_plan = (
    "🏠 7-DAY HOME WORKOUT PLAN (Muscle Hypertrophy):\n"
    "• Mon: Push Day (Chest/Shoulders/Triceps) - Decline Push-ups, Pike Push-ups, Chair Dips, Diamond Push-ups (3 sets x 12-15 reps).\n"
    "• Tue: Pull Day (Back/Biceps) - Doorframe Rows, Bed-sheet Pullups, Superman Extensions, Towel Bicep Curls (3 sets x 15 reps).\n"
    "• Wed: Leg Day (Quadriceps/Glutes) - Bulgarians Split Squats, Calf Raises, Walking Lunges, Wall Sits (4 sets x 15 reps).\n"
    "• Thu: Active Recovery - Light stretching and core work.\n"
    "• Fri: Full Body Strength Overload - Standard Push-ups, Bodyweight Squats, Pike Push-ups, Glute Bridges.\n"
    "• Sat: Core & Arm Definition - Plank-to-Pushups, Bicycle Crunches, Chair Dips, Plank Holds.\n"
    "• Sun: Complete Rest Day for Muscle Repair.\n\n"
    "🥗 MON-TO-SUN CALORIC SURPLUS DIET PLAN:\n"
    "• Mon-Wed:\n"
    "  - Breakfast: 3 Whole Eggs + 2 slices Whole Wheat Toast + 1 Banana with 2 tbsp Peanut Butter.\n"
    "  - Lunch: 200g Grilled Chicken/Paneer + 1.5 cups Boiled White Rice + 1 cup Dal.\n"
    "  - Snack: 1 cup Milk/Soy Milk + 1 scoop Whey Protein + 1 cup Oats blended.\n"
    "  - Dinner: 150g Fish or Tofu + 1 large Sweet Potato + Stir-fried veggies.\n"
    "• Thu-Sat:\n"
    "  - Breakfast: 1.5 cups Oatmeal cooked in milk + 1 tbsp honey + 15 chopped almonds.\n"
    "  - Lunch: 200g Fish or Mixed Bean Curry + 1.5 cups Brown Rice + 1 cup Curd.\n"
    "  - Snack: 2 boiled eggs (or 100g roasted paneer) + 1 cup Sprouts salad.\n"
    "  - Dinner: 200g Lean Chicken Breast or Soya Chunks + 2 Chapatis + Mixed vegetable sabzi.\n"
    "• Sun:\n"
    "  - Breakfast: Paneer Bhurji (150g) with 2 multi-grain Rotis.\n"
    "  - Lunch: Heavy Rice and Lentil Bowl (Khichdi) topped with 1 tbsp ghee + 150g chicken or tofu side.\n"
    "  - Snack: Handful of mixed dry fruits (cashews, raisins, walnuts).\n"
    "  - Dinner: High-protein paneer or chicken roll wrap using whole wheat flatbread."
)

endurance_plan = (
    "🏠 7-DAY HOME WORKOUT PLAN (Cardiovascular Endurance):\n"
    "• Mon: Aerobic Stamina - 40 minutes continuous jumping jacks, shadow boxing, and high knees at a steady, moderate pace.\n"
    "• Tue: Speed Intervals - 8 rounds: 1 minute maximum speed running-on-spot/burpees, 1 minute slow walk recovery.\n"
    "• Wed: Muscular Endurance Circuit - 4 rounds of 20 reps: Squats, Lunges, Pushups, Mountain Climbers (minimal rest).\n"
    "• Thu: Complete Rest & Muscle Length Restoration Stretching.\n"
    "• Fri: Tempo Pace Day - 30 mins continuous high-tempo bodyweight pacing (keep heart rate sustained at ~150 BPM).\n"
    "• Sat: Core Stamina & Agility - 45s Plank, 45s Side Plank, 45s Bird-Dog, 45s Skater Hops (Repeat for 4 rounds).\n"
    "• Sun: Active Long Distance Walk (45-60 mins outside).\n\n"
    "🥗 MON-TO-SUN CARB-SUSTAINED DIET PLAN:\n"
    "• Mon-Wed:\n"
    "  - Breakfast: 1.5 cups Oatmeal + 1 sliced Banana + 1 tbsp Flaxseeds + 2 boiled egg whites.\n"
    "  - Lunch: 1.5 cups Boiled Brown Rice + 1 cup Chickpea/Lentil Curry + Spinach Salad.\n"
    "  - Snack: 1 Whole Apple + 2 tbsp Peanut butter on rice cakes.\n"
    "  - Dinner: 150g Baked Salmon/Tofu + 1.5 cups Mashed Sweet Potatoes + Asparagus.\n"
    "• Thu-Sat:\n"
    "  - Breakfast: Fruit smoothie (Mango/Berry) with 1 scoop protein powder, rolled oats, and chia seeds.\n"
    "  - Lunch: Whole wheat pasta/Quinoa salad loaded with bell peppers, broccoli, and 150g shredded chicken/paneer.\n"
    "  - Snack: 1 cup Boiled Sprouts with chaat masala and lime.\n"
    "  - Dinner: 150g Grilled Chicken Breast or Soya Chunks + 1 cup boiled corn and peas + 2 Whole Wheat Chapatis.\n"
    "• Sun:\n"
    "  - Breakfast: 3 Egg Veggie Omelet + 2 slices of toasted brown bread.\n"
    "  - Lunch: Vegetable Biryani/Pulav cooked with high soy-chunks or paneer cubes + 1 bowl Raita.\n"
    "  - Snack: Handful of roasted Fox nuts (Makhana) + Green tea.\n"
    "  - Dinner: Vegetable and lentil hot soup stew with 1 slice of artisanal whole grain bread."
)

recommendations = []
for goal in goals:
    if goal == 0:
        recommendations.append(weight_loss_plan)
    elif goal == 1:
        recommendations.append(muscle_gain_plan)
    else:
        recommendations.append(endurance_plan)

df = pd.DataFrame({'age': ages, 'bmi': bmis, 'goal': goals, 'recommendation': recommendations})
df.to_csv('dataset.csv', index=False)

X = df[['age', 'bmi', 'goal']]
y = df['recommendation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained with complete detailed diet and home workout plans!")
