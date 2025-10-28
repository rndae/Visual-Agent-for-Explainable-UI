# Action Command Format

The VLM generates executable UI automation commands in a structured format.

## Command Syntax

### Click Command
```
Click(x, y, element_id, "element_description")
```
**Example:**
```
Click(414, 792, 12, "Save button")
```

### Type Command
```
Type(x, y, element_id, "field_name", "text_to_enter")
```
**Example:**
```
Type(411, 309, 4, "Name field", "Jose de la Rosa")
```

### Submit Command
```
Submit(x, y, element_id, "button_name")
```
**Example:**
```
Submit(414, 792, 12, "Save")
```

## Usage Example

```bash
# Step 1: Run OmniParser to detect UI elements
python run_omniparser.py images/form-example-2.png

# Step 2: Run VLM to generate action commands
./run_vlm.sh images/form-example-2.png --command "Fill form with: Jose de la Rosa, 242343111, josedelarosaroja@jose.com, 2444 pine st, Seal Beach, CA, 90740"
```

## Output Format

The VLM generates:
- **SUMMARY**: One-sentence description of the UI
- **ACTION PLAN**: Sequential list of executable commands

Example output:
```
SUMMARY:
New Customer form with fields for personal and address information.

ACTION PLAN:
Type(411, 309, 4, "Name", "Jose de la Rosa")
Type(440, 388, 5, "Phone Number", "242343111")
Type(591, 575, 15, "Address", "2444 pine st")
Type(490, 686, 14, "City", "Seal Beach")
Type(664, 686, 10, "State", "CA")
Type(440, 742, 13, "Code", "90740")
Click(414, 792, 12, "Save")
```

## Coordinate System

- Origin (0,0) is top-left corner
- Coordinates (x, y) represent center point of element
- Element IDs reference the OmniParser analysis

## Integration

These commands can be parsed and executed by automation frameworks like:
- OmniTool
- PyAutoGUI
- Selenium
- Playwright
- Custom automation scripts
