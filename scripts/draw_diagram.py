
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_debate_diagram(output_path='debate_diagram.png'):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Parameters
    num_agents = 3
    rounds = ['Round 0\n(Initial)', 'Round 1\n(Debate)', 'Round 2\n(Debate)', 'Round 3\n(Debate)']
    # Coordinates
    y_positions = [4.5, 3.3, 2.1, 0.9]
    x_positions = [1, 2, 3] # For Agents 1, 2, 3

    # Draw Question Box
    ax.add_patch(patches.FancyBboxPatch((1.5, 5.2), 1.0, 0.5, boxstyle="round,pad=0.1", ec="black", fc="#e6f2ff"))
    ax.text(2.0, 5.45, "Question", ha="center", va="center", fontsize=14, fontweight='bold')

    # Draw Rounds
    for r_idx, (r_name, y_pos) in enumerate(zip(rounds, y_positions)):
        # Draw Round Label
        ax.text(0.2, y_pos, r_name, ha="left", va="center", fontsize=12, fontweight='bold')
        
        for a_idx in range(num_agents):
            x_pos = x_positions[a_idx]
            
            # Determine color and style
            # Target is Round 1 (index 1), Agent 0 (first column)
            is_target = (r_idx == 1 and a_idx == 0)
            
            facecolor = "#ffcccc" if is_target else "#ccffcc"
            edgecolor = "red" if is_target else "black"
            linewidth = 2 if is_target else 1
            
            # Draw Agent Node (Circle)
            circle = patches.Circle((x_pos, y_pos), 0.25, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
            ax.add_patch(circle)
            
            # Label Agent
            ax.text(x_pos, y_pos, f"Agent {a_idx+1}", ha="center", va="center", fontsize=10)
            
            # Annotation for Target
            if is_target:
                ax.text(x_pos - 0.35, y_pos + 0.35, "Target\nDiff Temp", color="red", fontsize=10, ha="right")

            # Draw arrows from previous layer/question
            if r_idx == 0:
                # From Question
                ax.arrow(2.0, 5.2, x_pos - 2.0, (y_pos + 0.25) - 5.2 + 0.1, 
                         head_width=0.08, head_length=0.1, fc='black', ec='black', length_includes_head=True)
            else:
                # From previous round agents (fully connected)
                prev_y = y_positions[r_idx-1]
                for prev_a_idx in range(num_agents):
                    prev_x = x_positions[prev_a_idx]
                    # Draw thinner lines for mesh
                    ax.plot([prev_x, x_pos], [prev_y - 0.25, y_pos + 0.25], color="gray", linestyle="-", linewidth=0.5, zorder=0)

    # Draw Final Answer Box
    ax.add_patch(patches.FancyBboxPatch((1.5, -0.2), 1.0, 0.5, boxstyle="round,pad=0.1", ec="black", fc="#ffe6cc"))
    ax.text(2.0, 0.05, "Final Answer", ha="center", va="center", fontsize=14, fontweight='bold')
    
    # Arrows from last round
    for a_idx in range(num_agents):
        x_pos = x_positions[a_idx]
        y_pos = y_positions[-1]
        ax.arrow(x_pos, y_pos - 0.25, 2.0 - x_pos, 0.05 - (y_pos - 0.25) + 0.25, 
                 head_width=0.08, head_length=0.1, fc='black', ec='black', length_includes_head=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    draw_debate_diagram()
