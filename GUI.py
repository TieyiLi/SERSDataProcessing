from tkinter import *
import os
from tkinter import filedialog, ttk, scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from renishawWiRE import WDFReader
import numpy as np


class BlankGUI:
    def __init__(self, master, **kwargs):
        self.frm = Frame(master, **kwargs)
        self.frm.tkraise()


class ExportWDFGUI:

    def __init__(self, master):
        self.master = master

    def generate(self):
        self.frm = Toplevel(self.master, bd=30)
        self.frm.transient(self.master)
        self.frm.title('Export WDF')
        self.frm.geometry('1100x300+250+200')
        self.build_output()
        self.build_operations()

    def build_output(self):
        ttk.Separator(self.frm, orient='vertical').grid(row=0, rowspan=4, column=3, sticky=NS, padx=10)
        Label(self.frm, text='Running Progress', font=('Arial', 12, 'bold')).grid(row=0, column=4, sticky=N, padx=20)

        self.output = scrolledtext.ScrolledText(self.frm, height=12, width=40)
        self.output.grid(row=1, rowspan=3, column=4, sticky=N, padx=20)

    def build_operations(self):
        # input directory
        Label(self.frm, text='Input WDFs folder path:', font=('Arial', 12, 'bold')).grid(row=0, sticky=E)
        self.input_wdf_file_path_entry = Entry(self.frm, width=45)
        self.input_wdf_file_path_entry.grid(row=0, column=1, padx=20)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_input).grid(row=0, column=2)

        # export directory
        Label(self.frm, text='Input export folder path:', font=('Arial', 12, 'bold')).grid(row=1, sticky=E)
        self.input_export_path_entry = Entry(self.frm, width=45)
        self.input_export_path_entry.grid(row=1, column=1, padx=20)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_export).grid(row=1, column=2)

        # running or cancer
        button_run = Button(self.frm, text='Confirm and Run', height=1, width=15, command=self.execute)
        button_run.configure(font=('Arial', 12, 'bold'))
        button_run.grid(row=2, column=1, padx=10, pady=10)
        button_cancel = Button(self.frm, text='Cancel', height=1, width=15, command=self.terminate)
        button_cancel.configure(font=('Arial', 12, 'bold'))
        button_cancel.grid(row=3, column=1, padx=10)

        # pop window
        self.frm.tkraise()

    def browse_input(self):
        input_browser = \
            filedialog.askdirectory(initialdir='/', title='Select a directory')
        self.input_wdf_file_path_entry.insert(0, input_browser)

    def browse_export(self):
        export_browser = \
            filedialog.askdirectory(initialdir='/', title='Select a directory')
        self.input_export_path_entry.insert(0, export_browser)

    def execute(self):
        wdf_path = self.input_wdf_file_path_entry.get()
        export_path = self.input_export_path_entry.get()
        # add executing command
        print(wdf_path)
        print(export_path)

    def terminate(self):
        print('Exporting process terminated!')
        exit(-1)


class CrossValGUI():

    def __init__(self, master):
        self.master = master

    def generate(self):
        self.frm = Toplevel(self.master, bd=30)
        self.frm.transient(self.master)
        self.frm.title('Cross validation')
        self.frm.geometry('1100x550+250+200')
        self.build_operations()
        self.build_output()

    def browse_data_dict(self):
        input_browser = \
            filedialog.askopenfilename(initialdir='/', title='Select a file',
                                       filetype=[('Numpy', '*.npy')])
        self.input_data_dict_entry.insert(0, input_browser)

    def browse_label_dict(self):
        input_browser = \
            filedialog.askopenfilename(initialdir='/', title='Select a file',
                                       filetype=[('Numpy', '*.npy')])
        self.clustering_pred_entry.insert(0, input_browser)

    def write_output(self):
        return

    def build_output(self):
        ttk.Separator(self.frm, orient='vertical').grid(row=0, rowspan=12, column=3, sticky=NS, padx=10)
        Label(self.frm, text='Running Progress', font=('Arial', 12, 'bold')).grid(row=0, column=4, sticky=N, padx=20)

        self.output = scrolledtext.ScrolledText(self.frm, height=25, width=40)
        self.output.grid(row=1, rowspan=9, column=4, sticky=N, padx=20)

    def build_operations(self):
        # Input data dictionary
        Label(self.frm, text='Input data dictionary:', font=('Arial', 12, 'bold')).grid(row=0, sticky=E)
        self.input_data_dict_entry = Entry(self.frm, width=53)
        self.input_data_dict_entry.grid(row=0, column=1, padx=20)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_data_dict).grid(row=0, column=2)

        # Relabeling file path
        Label(self.frm, text='Input cluster prediction:', font=('Arial', 12, 'bold')).grid(row=2, sticky=E)
        self.clustering_pred_entry = Entry(self.frm, width=53, state=DISABLED)
        self.clustering_pred_entry.grid(row=2, column=1, padx=20)
        Button(self.frm, text='Open file', height=1, width=8, command=None).grid(row=2, column=2)

        # Use relabeling or not
        self.relabel_check_box_var = BooleanVar()

        # Relabeling
        Label(self.frm, text='Apply Relabeling:', font=('Arial', 12, 'bold')).grid(row=1, column=0, pady=10, sticky=E)

        # Relabeling checkbutton
        relabel_check_box = Checkbutton(self.frm, variable=self.relabel_check_box_var,
                                        command=self.activateCheck,
                                        font=('Arial', 12, 'bold'))
        relabel_check_box.grid(row=1, column=1, pady=10)

        # Cross-validation
        Label(self.frm, text='Cross-validation option:', font=('Arial', 12, 'bold')).grid(row=3, column=0, sticky=E)

        cv_var = StringVar()
        cv_var.set('Regular')
        cv_methods = {'Regular', 'Leave-map-out', 'Leave-sample-out', 'Leave-pair-of-sample-out'}
        cv_option = OptionMenu(self.frm, cv_var, *cv_methods)
        cv_option.grid(row=3, column=1, pady=10)
        cv_option.configure(font=('Arial', 12), width=30)

        Label(self.frm, text='')

        button_run = Button(self.frm, text='Confirm and Run', height=1, width=20)
        button_run.configure(font=('Arial', 12, 'bold'))
        button_run.grid(row=5, column=1, padx=10, pady=20)

        button_quit = Button(self.frm, text='Cancel', height=1, width=20)
        button_quit.configure(font=('Arial', 12, 'bold'))
        button_quit.grid(row=6, column=1)
        self.frm.tkraise()

    # Deactivate checkbox
    def activateCheck(self):
        if self.relabel_check_box_var.get():
            self.clustering_pred_entry.config(state=NORMAL)
        else:
            self.clustering_pred_entry.config(state=DISABLED)


class ClusterGUI():

    def __init__(self, master):
        self.master = master

    def generate(self):
        self.frm = Toplevel(self.master, bd=30)
        self.frm.transient(self.master)
        self.frm.title('Clustering')
        self.frm.geometry('1100x600+250+200')
        self.build_operations()
        self.build_output()

    def build_output(self):
        ttk.Separator(self.frm, orient='vertical').grid(row=0, rowspan=12, column=3, sticky=NS, padx=10)
        Label(self.frm, text='Running Progress', font=('Arial', 12, 'bold')).grid(row=0, column=4, sticky=N, padx=20)

        self.output = scrolledtext.ScrolledText(self.frm, height=25, width=40)
        self.output.grid(row=1, rowspan=9, column=4, sticky=N, padx=20)

    def build_operations(self):
        self.calculate_dm_widget()
        Label(self.frm, text='').grid(pady=10)
        self.generate_prediction_widget()
        self.frm.tkraise()

    def write_output(self):
        return


    def browse_data_dict(self):
        input_browser = \
            filedialog.askopenfilename(initialdir='/', title='Select a file',
                                       filetype=[('Numpy', '*.npy')])
        self.input_data_dict_entry.insert(0, input_browser)

    def browse_saveDM_dir(self):
        input_browser = \
            filedialog.askdirectory(initialdir='/', title='Select a directory')
        self.input_saveDMdir_entry.insert(0, input_browser)

    def browse_dm_dict(self):
        input_browser = \
            filedialog.askopenfilename(initialdir='/', title='Select a file',
                                       filetype=[('Numpy', '*.npy')])
        self.input_dm_entry.insert(0, input_browser)

    def browse_savepred_dir(self):
        input_browser = \
            filedialog.askdirectory(initialdir='/', title='Select a file')
        self.input_savepred_entry.insert(0, input_browser)

    def calculate_dm_widget(self):
        # input data dictionary
        Label(self.frm, text='Input data dictionary:', font=('Arial', 12, 'bold')).grid(column=0, sticky=E)
        self.input_data_dict_entry = Entry(self.frm, width=53)
        self.input_data_dict_entry.grid(row=0, column=1, padx=20)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_data_dict).grid(row=0, column=2)

        # save distance matrix label
        save_distance_matrix_label = Label(self.frm, text='Save distance matrix:', font=('Arial', 12, 'bold'))
        save_distance_matrix_label.grid(column=0, sticky=E)

        # save distance matrix checkbox
        self.save_DM_checkbox_var = BooleanVar()
        Checkbutton(self.frm, variable=self.save_DM_checkbox_var,
                    font=('Arial', 12, 'bold'),
                    command=self.activateCheckDM).grid(row=1, column=1)

        # distance matrix saving dir
        Label(self.frm, text='Input saving directory:', font=('Arial', 12, 'bold')).grid(column=0, sticky=E)
        self.input_saveDMdir_entry = Entry(self.frm, width=53, state=DISABLED)
        self.input_saveDMdir_entry.grid(row=2, column=1, padx=20)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_saveDM_dir).grid(row=2, column=2)

        # buttons
        button_run = Button(self.frm, text='Compute distance matrix', height=1, width=20)
        button_run.configure(font=('Arial', 12, 'bold'))
        button_run.grid(row=3, column=1, padx=10, pady=20)

        button_cancel = Button(self.frm, text='Cancel', height=1, width=20)
        button_cancel.configure(font=('Arial', 12, 'bold'))
        button_cancel.grid(row=4, column=1, padx=10)

    def generate_prediction_widget(self):

        Label(self.frm, text='')

        # input distance matrix label
        Label(self.frm, text='Input distance matrix:', font=('Arial', 12, 'bold')).grid(row=6, sticky=E)
        self.input_dm_entry = Entry(self.frm, width=53)
        self.input_dm_entry.grid(row=6, column=1, padx=20)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_dm_dict).grid(row=6, column=2)

        # save distance matrix label
        Label(self.frm, text='Save predictions:', font=('Arial', 12, 'bold')).grid(row=7, sticky=E)

        # saving distance matrix checkbox
        self.save_pred_checkbox_var = BooleanVar()
        Checkbutton(self.frm, variable=self.save_pred_checkbox_var,
                    font=('Arial', 12, 'bold'),
                    command=self.activateCheckpred).grid(row=7, column=1)

        # input saving prediction directory
        Label(self.frm, text='Input pred folder path:', font=('Arial', 12, 'bold')).grid(row=8, sticky=E)
        self.input_savepred_entry = Entry(self.frm, width=53, state=DISABLED)
        self.input_savepred_entry.grid(row=8, column=1, padx=20)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_savepred_dir).grid(row=8, column=2)

        # buttons
        button_run = Button(self.frm, text='Compute predictions', height=1, width=20)
        button_run.configure(font=('Arial', 12, 'bold'))
        button_run.grid(row=9, column=1, padx=10, pady=20)

        button_cancel = Button(self.frm, text='Cancel', height=1, width=20)
        button_cancel.configure(font=('Arial', 12, 'bold'))
        button_cancel.grid(row=10, column=1, padx=10)

    def activateCheckpred(self):
        if self.save_pred_checkbox_var.get():
            self.input_savepred_entry.config(state=NORMAL)
        else:
            self.input_savepred_entry.config(state=DISABLED)

    def activateCheckDM(self):
        if self.save_DM_checkbox_var.get():
            self.input_saveDMdir_entry.config(state=NORMAL)
        else:
            self.input_saveDMdir_entry.config(state=DISABLED)


class ReadDataDictGUI():

    def __init__(self, master):
        self.master = master

    def generate(self):
        self.frm = Toplevel(self.master, bd=30)
        self.frm.transient(self.master)
        self.frm.title('Read data dictionary')
        self.frm.geometry('1050x600+250+200')
        self.build_operations()
        self.build_output()

    def browse_spec_path(self):
        spec_path = filedialog.askdirectory(initialdir='/', title='Select a directory')
        self.input_specpath_entry.insert(0, spec_path)

    def browse_savedict_dir(self):
        data_dict_dir = filedialog.askdirectory(initialdir='/', title='Select a directory')
        self.input_savedict_entry.insert(0, data_dict_dir)

    def activateSavedict(self):
        if self.save_var.get():
            self.input_savedict_entry.config(state=NORMAL)
        else:
            self.input_savedict_entry.config(state=DISABLED)

    def write_output(self):
        self.output.insert('1.0', '')


    def build_output(self):
        ttk.Separator(self.frm, orient='vertical').grid(row=0, rowspan=12, column=3, sticky=NS)
        Label(self.frm, text='Running Progress', font=('Arial', 12, 'bold')).grid(row=0, column=4, sticky=N, padx=20)

        self.output = scrolledtext.ScrolledText(self.frm, height=30, width=40)
        self.output.grid(row=1, rowspan=10, column=4, sticky=N, padx=20)


    def build_operations(self):
        # input spectra directory
        Label(self.frm, text='Input data path:', font=('Arial', 12, 'bold')).grid(column=0, sticky=E)
        self.input_specpath_entry = Entry(self.frm, width=53)
        self.input_specpath_entry.grid(row=0, column=1)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_spec_path).grid(row=0, column=2, padx=10)

        # from *.txt or *.wdf directory
        Label(self.frm, text='Data type:', font=('Arial', 12, 'bold')).grid(row=1, column=0, sticky=E)

        self.file_type = StringVar()
        self.file_type.set('WDF')
        types = {'WDF', 'TXT'}
        file_type_menu = OptionMenu(self.frm, self.file_type, *types)
        file_type_menu.grid(row=1, column=1, pady=10)
        file_type_menu.configure(font=('Arial', 12), width=10)

        # input label dictionary manually
        Label(self.frm, text='Input label dict:', font=('Arial', 12, 'bold')).grid(row=2, sticky=E)
        Text(self.frm, height=3, width=40).grid(row=2, column=1, pady=10)

        # input group dictionary manually
        Label(self.frm, text='Input group dict:', font=('Arial', 12, 'bold')).grid(row=3, sticky=E)
        Text(self.frm, height=3, width=40).grid(row=3, column=1, pady=10)

        # interpolation checkbox
        self.interp_var = BooleanVar()
        Label(self.frm, text='Interpolation:', font=('Arial', 12, 'bold')).grid(row=4, sticky=E)
        Checkbutton(self.frm, variable=self.interp_var, font=('Arial', 12, 'bold')).grid(row=4, column=1)

        # substraction checkbox
        self.sub_var = BooleanVar()
        Label(self.frm, text='Subtraction:', font=('Arial', 12, 'bold')).grid(row=5, sticky=E)
        Checkbutton(self.frm, variable=self.sub_var, font=('Arial', 12, 'bold')).grid(row=5, column=1)

        # smooth checkbox
        self.smooth_var = BooleanVar()
        Label(self.frm, text='Smoothing:', font=('Arial', 12, 'bold')).grid(row=6, sticky=E)
        Checkbutton(self.frm, variable=self.smooth_var, font=('Arial', 12, 'bold')).grid(row=6, column=1)

        # normalization checkbox
        self.norm_var = BooleanVar()
        Label(self.frm, text='Normalization:', font=('Arial', 12, 'bold')).grid(row=7, sticky=E)
        Checkbutton(self.frm, variable=self.norm_var, font=('Arial', 12, 'bold')).grid(row=7, column=1)

        # save checkbox
        self.save_var = BooleanVar()
        Label(self.frm, text='Save dictionary:', font=('Arial', 12, 'bold')).grid(row=8, sticky=E)
        Checkbutton(self.frm, variable=self.save_var, font=('Arial', 12, 'bold'), command=self.activateSavedict).grid(row=8, column=1)

        # data dict saving directory
        Label(self.frm, text='Input saving directory:', font=('Arial', 12, 'bold')).grid(row=9, sticky=E)
        self.input_savedict_entry = Entry(self.frm, width=53, state=DISABLED)
        self.input_savedict_entry.grid(row=9, column=1, padx=10)
        Button(self.frm, text='Open file', height=1, width=8, command=self.browse_savedict_dir).grid(row=9, column=2)

        # run and cancal buttons
        button_run = Button(self.frm, text='Confirm and Run', height=1, width=20, command=None)
        button_run.configure(font=('Arial', 12, 'bold'))
        button_run.grid(row=10, column=1, padx=10, pady=20)

        button_cancel = Button(self.frm, text='Cancel', height=1, width=20, command=None)
        button_cancel.configure(font=('Arial', 12, 'bold'))
        button_cancel.grid(row=11, column=1, padx=10)

        self.frm.tkraise()


class GridVisualizer:

    def __init__(self, master, window_id, grid_coordinate):
        self.master = master
        self.window_id = window_id
        self.grid_coordinate = grid_coordinate

    def generate(self):
        pass

    def build_grid(self):
        return


class SpectrumVisualizer():

    def __init__(self, master, spectrum_file_path, window_id, data, data_size, grid_coordinate=None):
        self.window_id = window_id
        self.master = master
        self.spectrum_file_path = spectrum_file_path
        self.data = data
        self.data_size = data_size
        self.cur_index = 0
        self.grid_coordinate = grid_coordinate
        print(self.spectrum_file_path)

    def generate(self):
        self.frm = Toplevel(self.master, bd=30)
        self.frm.transient(self.master)
        self.frm.title('Spectra visualizer %d' % self.window_id)
        self.frm.geometry('800x800+500+80')

        self.frm.bind('<Left>', self.show_previous_plot)
        self.frm.bind('<Right>', self.show_next_plot)
        self.frm.bind('<Up>', self.show_up_plot)
        self.frm.bind('<Down>', self.show_down_plot)

        self.initialize_figure()
        self.build_plot()
        self.build_toolbar()
        self.build_operations()
        self.initialize_text()

        if not self.spectrum_file_path.endswith('.wdf') or self.data_size == 1:
            self.button_up.config(state=DISABLED)
            self.button_down.config(state=DISABLED)
        else:
            self.shape = [len(np.unique(self.grid_coordinate[0])), len(np.unique(self.grid_coordinate[1]))]

    def update_text_widget(self, text_widget, text):
        text_widget.delete('1.0', END)
        text_widget.insert('1.0', text)

    def show_next_plot(self, event=None):
        self.cur_index = (self.cur_index + 1) % self.data_size
        self.update_plotting(self.cur_index)

    def show_previous_plot(self, event=None):
        self.cur_index = (self.cur_index - 1) % self.data_size
        self.update_plotting(self.cur_index)

    def show_up_plot(self, event=None):
        if self.button_up['state'] == 'normal':
            self.cur_index = (self.cur_index - self.shape[0]) % self.data_size
            self.update_plotting(self.cur_index)

    def show_down_plot(self, event=None):
        if self.button_down['state'] == 'normal':
            self.cur_index = (self.cur_index + self.shape[0]) % self.data_size
            self.update_plotting(self.cur_index)

    def show_index_plot(self):
        next_index = int(self.index_text.get('1.0', END))
        if next_index < 0 or next_index >= self.data_size:
            messagebox.showerror(title='Index Error', message='Index out of range!')
        else:
            self.cur_index = next_index
            self.update_plotting(self.cur_index)

    def update_plotting(self, index):
        # get new spectrum
        intensity = self.data['data_matrix'][index]

        # plot new spectrum
        max_val = max(intensity)
        min_val = min(intensity)
        h = max_val - min_val
        self.ax.set_ylim(min_val - 0.1*h, max_val + 0.1*h)
        self.cur_line[0].set(ydata=intensity)
        self.fig_widge.draw()

        # update spectrum information
        self.update_spectrum_information(index)

    def update_spectrum_information(self, index):
        # update new spectrum info
        self.update_text_widget(self.index_text, index)

        if self.data['sample_name']:
            sample_name = self.data['sample_name'][index]
            self.update_text_widget(self.sample_name_text, sample_name)

        if self.data['map_index']:
            map_index = self.data['map_index'][index]
            self.update_text_widget(self.map_index_text, map_index)

        if self.data['group']:
            group_index = self.data['group'][index]
            self.update_text_widget(self.group_text, group_index)

        if self.data['file_name']:
            file_n = self.data['file_name'][index]
            self.update_text_widget(self.file_name_text, file_n)

        if  self.grid_coordinate:
            self.update_text_widget(self.x_coor, self.grid_coordinate[0][index])
            self.update_text_widget(self.y_coor, self.grid_coordinate[1][index])

    def initialize_figure(self):
        self.figure = Figure(figsize=(8, 3))
        self.ax = self.figure.add_subplot()
        self.ax.set_xlabel('Raman shift/cm^-1')
        self.ax.set_ylabel('Intensity')
        intensity = self.data['data_matrix'][0]
        self.cur_line = self.ax.plot(self.data['raman_shift'], intensity, 'r', lw=.8)
        max_val = max(intensity)
        min_val = min(intensity)
        h = max_val - min_val
        self.ax.set_ylim(min_val - 0.1 * h, max_val + 0.1 * h)

    def initialize_text(self):
        self.update_spectrum_information(0)
        if not self.data['file_name']:
            file_path_split = self.spectrum_file_path.split('/')
            self.update_text_widget(self.data_location_text, os.path.dirname(self.spectrum_file_path))
            self.update_text_widget(self.file_name_text, file_path_split[-1])
        else:
            self.update_text_widget(self.data_location_text, self.spectrum_file_path)
            self.update_text_widget(self.file_name_text, self.data['file_name'][0])


    def build_plot(self):
        # frame for figures and toolbar
        fig_frm = Frame(master=self.frm, width=800, height=400, relief=RIDGE, borderwidth=2)
        fig_frm.pack(anchor='w', expand=True)
        fig_frm.pack_propagate(False)
        self.fig_widge = FigureCanvasTkAgg(self.figure, master=fig_frm)
        self.fig_widge.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def build_toolbar(self):

        # get figure widget
        tool_frm = Frame(master=self.frm, width=800, height=40)
        tool_frm.pack(anchor='w', expand=True)
        tool_frm.pack_propagate(False)
        toolbar = NavigationToolbar2Tk(self.fig_widge, tool_frm)
        toolbar.pack(anchor='w', expand=True)
        toolbar.update()

    def build_operations(self):
        # frame for spectrum information
        info_frm = Frame(master=self.frm, width=650, height=250)
        info_frm.pack(anchor='w', expand=True)
        info_frm.pack_propagate(False)

        # previous and next buttons
        self.button_pre = Button(info_frm, text='\u25C4', height=1, width=4, font=('Arial', 12, 'bold'), command=self.show_previous_plot)
        self.button_pre.grid(row=1, column=3, sticky=NW)
        self.button_next = Button(info_frm, text='\u25BA', height=1, width=4, font=('Arial', 12, 'bold'), command=self.show_next_plot)
        self.button_next.grid(row=1, column=3, sticky=NE)

        # up and down buttons, only applicable for wdf maps
        self.button_up = Button(info_frm, text='\u25B2', height=1, width=4, font=('Arial', 12, 'bold'), command=self.show_up_plot)
        self.button_up.grid(row=0, column=3, sticky=N)
        self.button_down = Button(info_frm, text='\u25BC', height=1, width=4, font=('Arial', 12, 'bold'), command=self.show_down_plot)
        self.button_down.grid(row=1, column=3, sticky=N)

        # spectrum information labels
        Label(info_frm, text='Spectrum Information', font=('Arial', 13, 'bold')).grid(row=1, column=0, pady=10, sticky=E)

        # data index
        Label(info_frm, text='Index:', font=('Arial', 12)).grid(row=2, column=0, sticky=E)
        self.index_text = Text(info_frm, height=1, width=10)
        self.index_text.grid(row=2, column=1, sticky=W, padx=10)
        self.index_text.insert('1.0', 'Unknown')

        # go to certain position button
        button_goto = Button(info_frm, text='GoTo', height=1, width=5, command=self.show_index_plot)
        button_goto.grid(row=2, column=1, padx=30)

        # x coordinate
        Label(info_frm, text='X coordinate:', font=('Arial', 12)).grid(row=3, column=0, sticky=E)
        self.x_coor = Text(info_frm, height=1, width=60, borderwidth=0, background='#F3F3F3')
        self.x_coor.grid(row=3, column=1, columnspan=3, padx=10)
        self.x_coor.insert('1.0', 'Unknown')

        # y coordinate
        Label(info_frm, text='Y coordinate:', font=('Arial', 12)).grid(row=4, column=0, sticky=E)
        self.y_coor = Text(info_frm, height=1, width=60, borderwidth=0, background='#F3F3F3')
        self.y_coor.grid(row=4, column=1, columnspan=3, padx=10)
        self.y_coor.insert('1.0', 'Unknown')

        # sample name text
        Label(info_frm, text='Sample name:', font=('Arial', 12)).grid(row=5, column=0, sticky=E)
        self.sample_name_text = Text(info_frm, height=1, width=60, borderwidth=0, background='#F3F3F3')
        self.sample_name_text.grid(row=5, column=1, columnspan=3, padx=10)
        self.sample_name_text.insert('1.0', 'Unknown')

        # map index text
        Label(info_frm, text='Map index:', font=('Arial', 12)).grid(row=6, column=0, sticky=E)
        self.map_index_text = Text(info_frm, height=1, width=60, borderwidth=0, background='#F3F3F3')
        self.map_index_text.grid(row=6, column=1, columnspan=3, padx=10)
        self.map_index_text.insert('1.0', 'Unknown')

        # group index text
        Label(info_frm, text='Group index:', font=('Arial', 12)).grid(row=7, column=0, sticky=E)
        self.group_text = Text(info_frm, height=1, width=60, borderwidth=0, background='#F3F3F3')
        self.group_text.grid(row=7, column=1, columnspan=3, padx=10)
        self.group_text.insert('1.0', 'Unknown')

        # raman range text
        Label(info_frm, text='Raman range:', font=('Arial', 12)).grid(row=8, column=0, sticky=E)
        self.raman_range = Text(info_frm, height=1, width=60, borderwidth=0, background='#F3F3F3')
        self.raman_range.grid(row=8, column=1, columnspan=3, padx=10)
        self.raman_range.insert('1.0', 'Unknown')

        # data location
        Label(info_frm, text='Location:', font=('Arial', 12)).grid(row=9, column=0, sticky=E)
        self.data_location_text = Text(info_frm, height=1, width=60, borderwidth=0, background='#F3F3F3')
        self.data_location_text.grid(row=9, column=1, columnspan=3, padx=10)
        self.data_location_text.insert('1.0', 'Unknown')

        # file name
        Label(info_frm, text='File name:', font=('Arial', 12)).grid(row=10, column=0, sticky=E)
        self.file_name_text = Text(info_frm, height=1, width=60, borderwidth=0, background='#F3F3F3')
        self.file_name_text.grid(row=10, column=1, columnspan=3, padx=10)
        self.file_name_text.insert('1.0', 'Unknown')


class MultipleVisualizer():

    def __init__(self, master):
        self.master = master
        self.data_of_windows = {}
        self.cur_window_id = 1

    def callback(self, window):
        if messagebox.askokcancel(title='Close Visualizer', message='Do you really wish to quit?'):
            window.frm.destroy()
            del self.data_of_windows[window.window_id]
            print(len(self.data_of_windows))

    def new_visualizer(self, spectrum_path, data, data_size, grid_coordinate):
        print(len(self.data_of_windows))
        new_window = SpectrumVisualizer(self.master, spectrum_path, self.cur_window_id, data, data_size, grid_coordinate)
        new_window.generate()
        new_window.frm.protocol('WM_DELETE_WINDOW', lambda: self.callback(new_window))

    # wdf, single; wdf mapping; txt, single; txt directory
    def open_file(self):
        spectrum_browser = \
            filedialog.askopenfilename(initialdir='/',
                                   title='Select a file',
                                   filetype=[('WDF', '*.wdf'), ('TXT', '*.txt'), ('Dictionary', '*.npy')])

        input_data, data_size, grid_coordinate = {}, 0, None
        if spectrum_browser.endswith('.npy'):
            old_dict = np.load(spectrum_browser, allow_pickle=True).item()
            input_data = {key: old_dict[key] for key in ['data_matrix', 'sample_name', 'group', 'raman_shift',
                                                         'map_index']}
            data_size = len(input_data['group'])

        elif spectrum_browser.endswith('.wdf'):
            wdf_reading = WDFReader(spectrum_browser)
            input_data['raman_shift'] = wdf_reading.xdata

            spectra_collection = wdf_reading.spectra
            if spectra_collection.ndim == 3:
                ori_shape = spectra_collection.shape
                data_size = ori_shape[0] * ori_shape[1]
                input_data['data_matrix'] = spectra_collection.reshape(data_size, ori_shape[2])

            elif spectra_collection.ndim == 1:
                input_data['data_matrix'] = np.array([spectra_collection])
                data_size = 1

            input_data['sample_name'] = None
            input_data['group'] = None
            input_data['map_index'] = None
            input_data['file_name'] = None
            grid_coordinate = [wdf_reading.xpos, wdf_reading.ypos]

        else: # txt case
            spectrum = np.loadtxt(spectrum_browser)
            input_data = {'data_matrix': np.array([spectrum[:, 1]]), 'sample_name': None, 'group': None,
                          'raman_shift': spectrum[:, 0], 'map_index': None, 'file_name': None}
            data_size = 1

        self.data_of_windows[self.cur_window_id] = input_data
        self.new_visualizer(spectrum_browser, input_data, data_size, grid_coordinate)
        self.cur_window_id += 1

    def open_folder(self):
        spectrum_browser = \
            filedialog.askdirectory(initialdir='/',
                                   title='Select a folder')
        input_data, data_size, grid_coordinate = {}, 0, None
        map_index, data_matrix, file_name = \
            [], [], []
        pattern = re.compile(r'\\')
        for txt_str in os.listdir(spectrum_browser):
            if txt_str[-3:] == 'txt':
                file_name_split = pattern.split(spectrum_browser)
                map_index.append(file_name_split[-1])
                file_name.append(txt_str)
                spectrum = np.loadtxt(os.path.join(spectrum_browser, txt_str))
                data_matrix.append(spectrum[:, 1])
                raman_shift = spectrum[:, 0]
                data_size += 1

        input_data['data_matrix'] = np.array(data_matrix)
        input_data['sample_name'] = None
        input_data['map_index'] = map_index
        input_data['group'] = None
        input_data['raman_shift'] = raman_shift
        input_data['file_name'] = file_name

        self.data_of_windows[self.cur_window_id] = input_data
        self.new_visualizer(spectrum_browser, input_data, data_size, grid_coordinate)
        self.cur_window_id += 1




class CreateGUI(Tk):

    def __init__(self):
        Tk.__init__(self)
        self.title('SERS Analyses')
        self.geometry('1700x1000+200+50')
        self.container = Frame(self)
        self.container.grid()
        self.create_frame()
        self.create_menubar()

    def create_operation_menu(self, menu):
        operation_menu = Menu(menu, tearoff='off')
        menu.add_cascade(label='Operations', menu=operation_menu)
        operation_menu.add_command(label='Export WDF', command=self.frames[ExportWDFGUI].generate)
        operation_menu.add_command(label='Create data dictionary', command=self.frames[ReadDataDictGUI].generate)
        operation_menu.add_command(label='Clustering', command=self.frames[ClusterGUI].generate)
        operation_menu.add_command(label='Cross validation', command=self.frames[CrossValGUI].generate)

    def create_file_menu(self, menu):
        file_menu = Menu(menu, tearoff='off')
        menu.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open', command=self.frames[MultipleVisualizer].open_file)
        file_menu.add_command(label='Open txt folder', command=self.frames[MultipleVisualizer].open_folder)
        file_menu.add_command(label='Save', command=None)
        file_menu.add_command(label='Exit', command=exit)

    def create_menubar(self):
        menu = Menu(self)
        self.config(menu=menu)
        self.create_file_menu(menu=menu)
        self.create_operation_menu(menu)

    def create_frame(self):
        self.frames = {}
        self.frames[BlankGUI] = BlankGUI(self.container)
        self.frames[ClusterGUI] = ClusterGUI(self.container)
        self.frames[ExportWDFGUI] = ExportWDFGUI(self.container)
        self.frames[CrossValGUI] = CrossValGUI(self.container)
        self.frames[ReadDataDictGUI] = ReadDataDictGUI(self.container)
        self.frames[MultipleVisualizer] = MultipleVisualizer(self.container)


if __name__ == '__main__':
    gui = CreateGUI()
    gui.mainloop()
