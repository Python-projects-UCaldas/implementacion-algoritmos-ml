import tkinter as tk
import tkinter.font as tkFont
from algorithms import kmeans, hola
import os
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

class App:
    filename = None
    GLabel_920 = None
    GListBox_727 = None
    GButton_698 = None
    GButton_699 = None
    exceptionMessage = None

    def __init__(self, root):
        #setting title
        root.title("Implementación ML")
        #setting window size
        width=380
        height=300
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        self.GLabel_920=tk.Label(root)
        self.GLabel_920
        ft = tkFont.Font(family='Times',size=12)
        self.GLabel_920["font"] = ft
        self.GLabel_920["fg"] = "#333333"
        self.GLabel_920["justify"] = "center"
        self.GLabel_920["text"] = "Seleccione el algoritmo, cargue el \n archivo y pulse ejecutar"
        self.GLabel_920.place(x=20,y=20,width=300,height=41)

        self.GListBox_727=tk.Listbox(root)
        self.GListBox_727["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times',size=12)
        self.GListBox_727["font"] = ft
        self.GListBox_727["fg"] = "#333333"
        self.GListBox_727["justify"] = "center"
        self.GListBox_727.place(x=40,y=90,width=300,height=140)
        self.GListBox_727.insert(1, "KNN (K vecinos cercanos)")
        self.GListBox_727.insert(2, "Random forest")
        self.GListBox_727.insert(3, "KMEANS")
        self.GListBox_727.insert(4, "Dendograma (Cluster Jeráquicos)")
        self.GListBox_727.insert(5, "Naive Bayes")
        self.GListBox_727.insert(5, "Support Vector Machine")

        self.GButton_698=tk.Button(root)
        self.GButton_698["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=12)
        self.GButton_698["font"] = ft
        self.GButton_698["fg"] = "#000000"
        self.GButton_698["justify"] = "center"
        self.GButton_698["text"] = "Seleccionar archivo"
        self.GButton_698.place(x=40,y=250,width=130,height=25)
        self.GButton_698["command"] = self.GButton_698_command

        self.GButton_699=tk.Button(root)
        self.GButton_699["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=12)
        self.GButton_699["font"] = ft
        self.GButton_699["fg"] = "#000000"
        self.GButton_699["justify"] = "center"
        self.GButton_699["text"] = "Ejecutar"
        self.GButton_699.place(x=220,y=250,width=130,height=25)
        self.GButton_699["command"] = self.Execute

    
    def GButton_698_command(self):
        filetypes = (
        ('Archivos csv', '*.csv'),
        ('All files', '*.*')
        )

        self.filename = fd.askopenfilename(
            title='Abrir archivo',
            initialdir=os.getcwd(),
            filetypes=filetypes)

        showinfo(
            title='Archivo seleccionado',
            message=self.filename
        )

    def Execute(self):
        try:
            if self.filename == None:
                self.exceptionMessage = 'Debes seleccionar un archivo'
                print(self.GListBox_727.curselection())
                raise Exception()
            if len(self.GListBox_727.curselection()) == 0:
                self.exceptionMessage = 'Debes seleccionar un algoritmo'
                raise Exception()
            if self.GListBox_727.curselection()[0] == 0:
                hola()
            elif self.GListBox_727.curselection()[0] == 1:
                print("Random forest")
            elif self.GListBox_727.curselection()[0] == 2:
                kmeans(self.filename)
            elif self.GListBox_727.curselection()[0] == 3:
                print("Dendograma (Cluster Jeráquicos)")
            elif self.GListBox_727.curselection()[0] == 4:
                print("Naive Bayes")
            elif self.GListBox_727.curselection()[0] == 5:
                print("Support Vector Machine")
        except:
            showinfo(
            title='Alerta',
            message=self.exceptionMessage
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()