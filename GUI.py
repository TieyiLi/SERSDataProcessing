from tkinter import *
from Tools.PrepareData import PrepareData
from Tools.ExportWDF import export_wdf
import json, os
import numpy as np
from Tools.Utils import Utils
from Tools.AffinityMatrix import distance_matrix
from Tools.AdditionalProcessing import AdditionalProcessing
from sklearn.cluster import AgglomerativeClustering
from Tools.CrossValidation import regular_CV, leave_maps_out_CV, leave_one_sample_out_CV, leave_pair_of_samples_out_CV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

PD = PrepareData()
uti = Utils()
addprep = AdditionalProcessing()


class ExportWDFGUI:

    def __init__(self, master, **kwargs):
        self.frm = Frame(master, **kwargs)
        self.frm.grid(row=0, column=0, padx=30, pady=20)
        self.frm.grid_propagate(False)

        input_wdf_file_path_label = Label(self.frm, text='Input WDFs folder path:', font=('Arial', 12, 'bold'))
        input_wdf_file_path_label.grid(sticky=E)
        self.input_wdf_file_path_entry = Entry(self.frm, width=40)
        self.input_wdf_file_path_entry.grid(row=0, column=1, sticky=W, pady=10)
        self.input_wdf_file_path_entry.grid_propagate(False)

        input_export_path_label = Label(self.frm, text='Input export folder path:', font=('Arial', 12, 'bold'))
        input_export_path_label.grid(sticky=E)
        self.input_export_path_entry = Entry(self.frm, width=40)
        self.input_export_path_entry.grid(row=1, column=1, sticky=W, pady=10)
        self.input_export_path_entry.grid_propagate(False)

        button_run = Button(self.frm, text='RUN', height=3, width=8, command=self.execute)
        button_run.configure(font=('Arial', 12))
        button_run.grid(padx=10, pady=30, sticky=E)

        button_cancel = Button(self.frm, text='CANCEL', height=3, width=8, command=self.terminate)
        button_cancel.configure(font=('Arial', 12))
        button_cancel.grid(row=2, column=1, pady=30)

    def execute(self):
        wdf_path = self.input_wdf_file_path_entry.get()
        export_path = self.input_export_path_entry.get()
        export_wdf(wdf_path, export_path)

    def terminate(self):
        print('Exporting process terminated!')
        exit(-1)


class SelectSpecGUI:

    def __init__(self, master, **kwargs):
        self.frm = Frame(master, **kwargs)
        self.frm.grid(row=0, column=0, padx=30, pady=20)
        self.frm.grid_propagate(False)

        input_specpath_label = Label(self.frm, text='Input raw spec path:', font=('Arial', 12, 'bold'))
        input_specpath_label.grid(sticky=E)
        self.input_specpath_entry = Entry(self.frm, width=40)
        self.input_specpath_entry.grid(row=0, column=1, sticky=W)
        self.input_specpath_entry.grid_propagate(False)

        self.save_checkbox_var = BooleanVar()
        save_check_box = Checkbutton(self.frm, text='Save qualified spec', variable=self.save_checkbox_var,
                                     command=self.activateEntry, font=('Arial', 12, 'bold'))
        save_check_box.grid(column=1, sticky=W)


        input_savepath_label = Label(self.frm, text='Input saving path:', font=('Arial', 12, 'bold'))
        input_savepath_label.grid(column=0, sticky=E)
        self.input_savepath_entry = Entry(self.frm, state=DISABLED, width=40)
        self.input_savepath_entry.grid(row=2, column=1, sticky=W)
        self.input_savepath_entry.grid_propagate(False)

        input_number_label = Label(self.frm, text='Input num per map:', font=('Arial', 12, 'bold'))
        input_number_label.grid(column=0, sticky=E)
        self.input_number_entry = Entry(self.frm, width=40)
        self.input_number_entry.grid(row=3, column=1, sticky=W)
        self.input_number_entry.grid_propagate(False)

        button_run = Button(self.frm, text='RUN', height=3, width=8, command=self.execute)
        button_run.configure(font=('Arial', 12))
        button_run.grid(padx=10, pady=30, sticky=E)

        button_cancel = Button(self.frm, text='CANCEL', height=3, width=8, command=self.terminate)
        button_cancel.configure(font=('Arial', 12))
        button_cancel.grid(row=4, column=1, pady=30)

    def activateEntry(self):
        if self.save_checkbox_var.get():
            self.input_savepath_entry.config(state=NORMAL)
        else:
            self.input_savepath_entry.config(state=DISABLED)

    def execute(self):
        data_from = self.input_specpath_entry.get()
        data_to = self.input_savepath_entry.get()
        n_each_map = int(self.input_number_entry.get())
        copy = self.save_checkbox_var.get()
        PD.ranking_selection(data_from, data_to, n_each_map, copy=copy)

    def terminate(self):
        print('Process terminated!')
        exit(-1)


class ReadDataDictGUI:

    def __init__(self, master, **kwargs):
        self.frm = Frame(master, **kwargs)
        self.frm.grid(row=0, column=0, padx=30, pady=20)
        self.frm.grid_propagate(False)

        input_specpath_label = Label(self.frm, text='Input qualified spec path:', font=('Arial', 12, 'bold'))
        input_specpath_label.grid(column=0, sticky=E)
        self.input_specpath_entry = Entry(self.frm, width=40)
        self.input_specpath_entry.grid(row=0, column=1, sticky=W)
        self.input_specpath_entry.grid_propagate(False)

        input_labeldict_label = Label(self.frm, text='Input label dict:', font=('Arial', 12, 'bold'))
        input_labeldict_label.grid(column=0, sticky=E)
        self.input_labeldict_entry = Entry(self.frm, width=40)
        self.input_labeldict_entry.grid(row=1, column=1, sticky=W)
        self.input_labeldict_entry.grid_propagate(False)

        input_neg_label = Label(self.frm, text='Input negative group:', font=('Arial', 12, 'bold'))
        input_neg_label.grid(column=0, sticky=E)
        self.input_neg_entry = Entry(self.frm, width=40)
        self.input_neg_entry.grid(row=2, column=1, sticky=W)
        self.input_neg_entry.grid_propagate(False)

        input_pos_label = Label(self.frm, text='Input positive group:', font=('Arial', 12, 'bold'))
        input_pos_label.grid(column=0, sticky=E)
        self.input_pos_entry = Entry(self.frm, width=40)
        self.input_pos_entry.grid(row=3, column=1, sticky=W)
        self.input_pos_entry.grid_propagate(False)

        input_dir_label = Label(self.frm, text='Input saving path:', font=('Arial', 12, 'bold'))
        input_dir_label.grid(column=0, sticky=E)
        self.input_dir_entry = Entry(self.frm, state=DISABLED, width=40)
        self.input_dir_entry.grid(row=4, column=1, sticky=W)
        self.input_dir_entry.grid_propagate(False)

        input_filename_label = Label(self.frm, text='Input file name:', font=('Arial', 12, 'bold'))
        input_filename_label.grid(column=0, sticky=E)
        self.input_filename_entry = Entry(self.frm, state=DISABLED, width=40)
        self.input_filename_entry.grid(row=5, column=1, sticky=W)
        self.input_filename_entry.grid_propagate(False)

        self.interp_var = BooleanVar()
        interp_checkbox = Checkbutton(self.frm, text='Interpolation',
                                      variable=self.interp_var, font=('Arial', 12, 'bold'))
        interp_checkbox.grid(sticky=W)

        self.sub_var = BooleanVar()
        sub_checkbox = Checkbutton(self.frm, text='Subtraction', variable=self.sub_var, font=('Arial', 12, 'bold'))
        sub_checkbox.grid(row=6, column=1, sticky=W)

        self.smooth_var = BooleanVar()
        smooth_checkbox = Checkbutton(self.frm, text='Smoothing', variable=self.smooth_var, font=('Arial', 12, 'bold'))
        smooth_checkbox.grid(sticky=W)

        self.norm_var = BooleanVar()
        norm_checkbox = Checkbutton(self.frm, text='Normalization', variable=self.norm_var, font=('Arial', 12, 'bold'))
        norm_checkbox.grid(row=7, column=1, sticky=W)

        self.save_var = BooleanVar()
        norm_checkbox = Checkbutton(self.frm, text='Save Dict', variable=self.save_var, command=self.activateEntry,
                                    font=('Arial', 12, 'bold'))
        norm_checkbox.grid(sticky=W)

        button_run = Button(self.frm, text='RUN', height=1, width=12, command=self.execute)
        button_run.configure(font=('Arial', 12))
        button_run.grid(padx=10, pady=5, sticky=E)

        button_cancel = Button(self.frm, text='CANCEL', height=1, width=12, command=self.terminate)
        button_cancel.configure(font=('Arial', 12))
        button_cancel.grid(row=9, column=1)

    def activateEntry(self):
        if self.save_var.get():
            self.input_dir_entry.config(state=NORMAL)
            self.input_filename_entry.config(state=NORMAL)
        else:
            self.input_dir_entry.config(state=DISABLED)
            self.input_filename_entry.config(state=DISABLED)

    def execute(self):
        data_path = self.input_specpath_entry.get()
        label_dict = json.loads(self.input_labeldict_entry.get())
        neg = json.loads(self.input_neg_entry.get())
        pos = json.loads(self.input_pos_entry.get())
        group_dict = {0: neg, 1: pos}
        interp = self.interp_var.get()
        sub = self.sub_var.get()
        smooth = self.smooth_var.get()
        norm = self.norm_var.get()
        dire = self.input_dir_entry.get()
        filename = self.input_filename_entry.get()
        save = self.save_var.get()

        data_dict = PD.read_data(data_path, label_dict, group_dict, interp)
        if save:
            PD.save(data_dict, dire, filename, sub, smooth, norm)

    def terminate(self):
        print('Process terminated!')
        exit(-1)


class CrossValGUI:

    def __init__(self, master, **kwargs):
        self.frm = Frame(master, **kwargs)
        self.frm.grid(row=0, column=0, padx=30, pady=20)
        self.frm.grid_propagate(False)

        input_data_dict_label = Label(self.frm, text='Input data dictionary:', font=('Arial', 12, 'bold'))
        input_data_dict_label.grid(column=0, sticky=E, padx=30)
        self.input_data_dict_entry = Entry(self.frm, width=40)
        self.input_data_dict_entry.grid(row=0, column=1, sticky=W, pady=10)
        self.input_data_dict_entry.grid_propagate(False)

        self.relabel_check_box_var = BooleanVar()

        clustering_pred_label = Label(self.frm, text='Input cluster prediction:', font=('Arial', 12, 'bold'))
        clustering_pred_label.grid(row=2, sticky=E, padx=30)
        self.clustering_pred_entry = Entry(self.frm, state=DISABLED, width=40)
        self.clustering_pred_entry.grid(row=2, column=1, sticky=W, pady=10)
        self.clustering_pred_entry.grid_propagate(False)

        def activateCheck():
            if self.relabel_check_box_var.get():
                self.clustering_pred_entry.config(state=NORMAL)
            else:
                self.clustering_pred_entry.config(state=DISABLED)

        relabel_check_box = Checkbutton(self.frm, text='Apply relabeling', variable=self.relabel_check_box_var,
                                        command=activateCheck,
                                        font=('Arial', 12, 'bold'))
        relabel_check_box.grid(row=1, column=1, sticky=W, pady=10)

        cv_label = Label(self.frm, text='Cross-validation option:', font=('Arial', 12, 'bold'))
        cv_label.grid(row=3, sticky=E, padx=30)

        self.cv_var = StringVar()
        self.cv_var.set('regular')
        cv_methods = {'regular', 'leave-maps-out', 'leave-one-sample-out', 'leave-pair-of-samples-out'}
        cv_option = OptionMenu(self.frm, self.cv_var, *cv_methods)
        cv_option.grid(row=3, column=1, sticky=W, pady=10)
        cv_option.configure(font=('Arial', 12))

        button_run = Button(self.frm, text='RUN', height=3, width=15, command=self.execute)
        button_run.configure(font=('Arial', 12))
        button_run.grid(row=4, pady=30, sticky=E)

        button_quit = Button(self.frm, text='EXIT', height=3, width=15, command=self.terminate)
        button_quit.configure(font=('Arial', 12))
        button_quit.grid(row=4, column=1, sticky=W)

    def execute(self):
        data_dict_path = self.input_data_dict_entry.get()
        data_dict = addprep.input_data(data_dict_path, output='dict')
        data_matrix = data_dict['data_matrix']
        label = data_dict['label']
        group = data_dict['group']
        # label_dict = data_dict['label_dict']
        group_dict = data_dict['group_dict']
        # raman_shift = data_dict['raman_shift']
        map_index = data_dict['map_index']

        smoothed_data_matrix = uti.smoothing(data_matrix)
        clf_svc = SVC(C=5.0, gamma='scale')

        if self.relabel_check_box_var.get():
            cluster_pred = np.load(self.clustering_pred_entry.get())
            group = addprep.relabel_by_purity(cluster_pred, group)

        cv_method = self.cv_var.get()
        if cv_method == 'regular':
            splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
            regular_CV(clf_svc, smoothed_data_matrix, group, splitter, output='accuracy')
        elif cv_method == 'leave-maps-out':
            leave_maps_out_CV(clf_svc, smoothed_data_matrix, group, map_index, k=5)
        elif cv_method == 'leave-one-sample-out':
            pred, true, score = \
                leave_one_sample_out_CV(clf_svc, smoothed_data_matrix, group, label, group_dict, map_index)
            addprep.manual_roc(true, score)
        elif cv_method == 'leave-pair-of-samples-out':
            pred, true, score = \
                leave_pair_of_samples_out_CV(clf_svc, smoothed_data_matrix, group, label, group_dict, map_index)
            addprep.manual_roc(true, score)

    def terminate(self):
        print('Process terminated!')
        exit(-1)


class ClusterGUI:

    def __init__(self, master, **kwargs):
        self.frm = Frame(master, **kwargs)
        self.frm.grid(row=0, column=0, padx=30, pady=20)
        self.frm.grid_propagate(False)
        self.calculate_dm_widget()
        self.generate_prediction_widget()

    def calculate_dm_widget(self):
        input_data_dict_label = Label(self.frm, text='Input data dictionary:', font=('Arial', 12, 'bold'))
        input_data_dict_label.grid(column=0, sticky=E, padx=30)
        self.input_data_dict_entry = Entry(self.frm, width=40)
        self.input_data_dict_entry.grid(row=0, column=1, sticky=W)
        self.input_data_dict_entry.grid_propagate(False)

        self.save_DM_checkbox_var = BooleanVar()
        save_DM_checkbox = Checkbutton(self.frm, text='Save DM', variable=self.save_DM_checkbox_var,
                                       command=self.activateEntry_top, font=('Arial', 12, 'bold'))
        save_DM_checkbox.grid(row=1, column=1, sticky=W)

        input_saveDMdir_label = Label(self.frm, text='Input DM saving path:', font=('Arial', 12, 'bold'))
        input_saveDMdir_label.grid(column=0, sticky=E, padx=30)
        self.input_saveDMdir_entry = Entry(self.frm, state=DISABLED, width=40)
        self.input_saveDMdir_entry.grid(row=2, column=1, sticky=W)
        self.input_saveDMdir_entry.grid_propagate(False)

        input_DMname_label = Label(self.frm, text='Input DM file name:', font=('Arial', 12, 'bold'))
        input_DMname_label.grid(column=0, sticky=E, padx=30)
        self.input_DMname_entry = Entry(self.frm, state=DISABLED, width=40)
        self.input_DMname_entry.grid(row=3, column=1, sticky=W)
        self.input_DMname_entry.grid_propagate(False)

        button_run = Button(self.frm, text='Compute DM', height=1, width=12, command=self.execute_top)
        button_run.configure(font=('Arial', 12))
        button_run.grid(padx=10, sticky=E)

        button_cancel = Button(self.frm, text='CANCEL', height=1, width=12, command=self.terminate)
        button_cancel.configure(font=('Arial', 12))
        button_cancel.grid(row=4, column=1)

    def generate_prediction_widget(self):
        input_DM_label = Label(self.frm, text='Input DM:', font=('Arial', 12, 'bold'))
        input_DM_label.grid(column=0, sticky=E, padx=30)
        self.input_DM_entry = Entry(self.frm, width=40)
        self.input_DM_entry.grid(row=5, column=1, sticky=W)
        self.input_DM_entry.grid_propagate(False)

        self.save_pred_checkbox_var = BooleanVar()
        relabel_check_box = Checkbutton(self.frm, text='Save pred', variable=self.save_pred_checkbox_var,
                                        command=self.activateEntry_bottom, font=('Arial', 12, 'bold'))
        relabel_check_box.grid(row=6, column=1, sticky=W)

        input_savepred_label = Label(self.frm, text='Input pred saving path:', font=('Arial', 12, 'bold'))
        input_savepred_label.grid(column=0, sticky=E, padx=30)
        self.input_savepred_entry = Entry(self.frm, state=DISABLED, width=40)
        self.input_savepred_entry.grid(row=7, column=1, sticky=W)
        self.input_savepred_entry.grid_propagate(False)

        input_predname_label = Label(self.frm, text='Input pred file name:', font=('Arial', 12, 'bold'))
        input_predname_label.grid(column=0, sticky=E, padx=30)
        self.input_predname_entry = Entry(self.frm, state=DISABLED, width=40)
        self.input_predname_entry.grid(row=8, column=1, sticky=W)
        self.input_predname_entry.grid_propagate(False)

        button_run = Button(self.frm, text='Compute pred', height=1, width=12, command=self.execute_bottom)
        button_run.configure(font=('Arial', 12))
        button_run.grid(padx=10, sticky=E)

        button_cancel = Button(self.frm, text='CANCEL', height=1, width=12, command=self.terminate)
        button_cancel.configure(font=('Arial', 12))
        button_cancel.grid(row=9, column=1)

    def execute_top(self):
        print('Running...')
        data_dict_path = self.input_data_dict_entry.get()
        save_dm = self.save_DM_checkbox_var.get()
        save_dm_dir = self.input_saveDMdir_entry.get()
        dm_filename = self.input_DMname_entry.get()

        data_dict = addprep.input_data(data_dict_path, output='dict')
        data_matrix = data_dict['data_matrix']
        X = uti.smoothing(data_matrix)
        dist_mtx = distance_matrix(X)
        if save_dm:
            PATH = os.path.join(save_dm_dir, dm_filename)
            np.save(PATH, dist_mtx)
        print('Finished distance matrix!')

    def execute_bottom(self):
        print('Running...')
        dm_path = self.input_DM_entry.get()
        save_pred = self.save_pred_checkbox_var.get()
        save_pred_dir = self.input_savepred_entry.get()
        pred_filename = self.input_predname_entry.get()

        dist_mtx = np.load(dm_path)
        hca = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete', distance_threshold=0.125)
        pred = hca.fit_predict(dist_mtx)
        if save_pred:
            PATH = os.path.join(save_pred_dir, pred_filename)
            np.save(PATH, pred)
        print('Finished predicting!')

    def activateEntry_top(self):
        if self.save_DM_checkbox_var.get():
            self.input_saveDMdir_entry.config(state=NORMAL)
            self.input_DMname_entry.config(state=NORMAL)
        else:
            self.input_saveDMdir_entry.config(state=DISABLED)
            self.input_DMname_entry.config(state=DISABLED)

    def activateEntry_bottom(self):
        if self.save_pred_checkbox_var.get():
            self.input_savepred_entry.config(state=NORMAL)
            self.input_predname_entry.config(state=NORMAL)
        else:
            self.input_savepred_entry.config(state=DISABLED)
            self.input_predname_entry.config(state=DISABLED)

    def terminate(self):
        print('Process terminated!')
        exit(-1)


class CreateGUI(Tk):

    def __init__(self):
        Tk.__init__(self)
        self.title('SERS Analyses')
        self.geometry('600x400+700+250')
        self.container = Frame(self)
        self.container.grid(padx=20, pady=10)
        self.create_frame()
        self.create_menubar()

    def create_menubar(self):
        menu = Menu(self)
        self.config(menu=menu)

        operation_menu = Menu(menu, tearoff='off')
        menu.add_cascade(label='Operations', menu=operation_menu)
        operation_menu.add_command(label='Export WDF', command=lambda: self.show_frame(ExportWDFGUI))
        operation_menu.add_separator()
        operation_menu.add_command(label='Select spectra', command=lambda: self.show_frame(SelectSpecGUI))
        operation_menu.add_separator()
        operation_menu.add_command(label='Create data dictionary', command=lambda: self.show_frame(ReadDataDictGUI))
        operation_menu.add_separator()
        operation_menu.add_command(label='Clustering', command=lambda: self.show_frame(ClusterGUI))
        operation_menu.add_separator()
        operation_menu.add_command(label='Cross validation', command=lambda: self.show_frame(CrossValGUI))

    def create_frame(self):
        self.frames = {}
        self.frames[ClusterGUI] = ClusterGUI(self.container, width=500, height=300, relief=GROOVE, borderwidth=3)
        self.frames[ExportWDFGUI] = ExportWDFGUI(self.container, width=500, height=300, relief=GROOVE, borderwidth=3)
        self.frames[CrossValGUI] = CrossValGUI(self.container, width=500, height=300, relief=GROOVE, borderwidth=3)
        self.frames[ReadDataDictGUI] = ReadDataDictGUI(self.container, width=500, height=300, relief=GROOVE, borderwidth=3)
        self.frames[SelectSpecGUI] = SelectSpecGUI(self.container, width=500, height=300, relief=GROOVE, borderwidth=3)
        self.show_frame(ExportWDFGUI)

    def show_frame(self, frm):
        frame = self.frames[frm]
        frame.frm.tkraise()

gui = CreateGUI()
gui.mainloop()
