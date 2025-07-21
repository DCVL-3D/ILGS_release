# 🪄 3D Editing Examples

This guide demonstrates how to perform common 3D editing tasks such as object removal, color modification, and resizing using our tools.

### **Prerequisite: Training**

Before you can perform any edits, you must first complete the training process as described in the [Implementation Guide](Implementation.md). Once training is finished, save the output model file, which you will then use for editing.

---

## 🗑️ 1. 3D Object Removal

To remove an object from the scene, you can specify it with a simple text query.

Run the editing script and provide a text prompt for the object you wish to delete.

Example: figurines dataset

```
python edit_removal.py -m output/lerf/figurines
```

After running the command, you will be prompted to enter your query:
```
Enter the text query to remove: a description of the object to remove
```

## 🎨 2. 3D Object Recolor

This feature allows you to select an object via a text query and change its color from a predefined list.

First, provide a text prompt for the target object, then select a color from the presented options.

Example: figurines dataset

```
python edit_recolor.py -m output/lerf/figurines
```
The script will then ask for the object to recolor and the desired color:
```
Enter the text query of the object to recolor: a description of the object to recolor

Please choose a color to apply:
  1: Red
  2: Green
  3: Blue
  4: Yellow
  5: Magenta
  6: Cyan
  7: Orange
  8: Purple
  9: White
  10: Black
Enter a number (1-10): 
```

## 📏 3. 3D Object Resizing

You can resize a specific object by identifying it with a text query and providing a numerical scale factor. A factor less than 1.0 will shrink the object, while a factor greater than 1.0 will enlarge it.

Example: teatime dataset

```
python edit_resizing.py -m output/lerf/teatime
```
The script will then ask for the object to resize and the desired scale factor
```
Enter the text query of the object to resize: a description of the object to resize

Enter the scale factor (e.g., 0.5 to shrink, 1.5 to enlarge):
```
