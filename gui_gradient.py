import tkinter as tk
from tkinter import ttk

def gui_table(df, method):
    root = tk.Tk()
    root.title(f"Iteraciones del método {method}")
    tree = ttk.Treeview(root)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col)

    for row in df.itertuples(index=False):
        tree.insert("", "end", values=row)

    tree.pack(expand=True, fill='both')

    root.mainloop()