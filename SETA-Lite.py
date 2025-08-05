#!/usr/bin/env python
# coding: utf-8

# # Load, visualise and reconstruct data with DiondoDataReader
# This how-to shows how to use the `DiondoDataReader` 

# In[1]:


#   The University of Southampton 
#   μ-VIS X-ray Imaging Centre 
#   Authored by:   Kubra Kumrular (RKK)

#   This file contains 3 Mode reconstructions:
#   ''' Modes:
#       1 = Load only 360 projections (equally spaced from all available projections)
#       2 = Load entire dataset (all projections)
#       3 = Load only central slices from each of all projections (2D slice)'''


# ### Import libraries

# In[2]:


import datetime
import os

log_messages = []  # Global list to collect log messages

def save_log(log_message, log_file_path=None):
    """
    Save a log message to a specified log file with a timestamp.
    If no path is provided, it accumulates messages in the global list.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] {log_message}"

    # If the log file path is not defined, accumulate the message
    if log_file_path is None:
        log_messages.append(message)
        return

    # Save all accumulated messages first
    with open(log_file_path, "a") as log_file:
        for msg in log_messages:
            log_file.write(msg + "\n")
        # Clear the global list after saving
        log_messages.clear()
        # Save the current message as well
        log_file.write(message + "\n")
    print(f" Log saved to: {log_file_path}")

# Collect the initial log message
save_log("Log started for Diondo reconstruction process.")


# In[3]:


# external libs
import time
import xml.etree.ElementTree as ET
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import logging
import tifffile as tiff
import gc  # module for the cleaning the memory
from tkinter import Tk, filedialog, simpledialog, messagebox
from tkinter import Tk, Label, Button, IntVar, Radiobutton, Checkbutton, filedialog, simpledialog, messagebox,font
import threading
from joblib import Parallel, delayed # parallel uploding 
import multiprocessing

# cil imports
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.framework import ImageData, ImageGeometry
from cil.processors import Slicer, AbsorptionTransmissionConverter, TransmissionAbsorptionConverter,CentreOfRotationCorrector,Binner,Normaliser, Padder
from cil.utilities.display import show2D, show_geometry # show1D,
from cil.recon import FDK
from cil.plugins.tigre import FBP
from cil.io import TIFFWriter, RAWFileWriter


# In[4]:


class DiondoDataReader(object):
    """
    Optimized reader for Diondo XML metadata and projection data (RAW format)

    Modes:
    1 = Load 360 projections only (evenly spaced from the available all projections)
    2 = Load full dataset (all of projections)
    3 = Load only central slices from each of the all projections (2D slice)

    Binning:
    - Applied only for Mode 1 & Mode 2 (3D reconstructions)
    - User can enable binning via `apply_binning=True` and set `binning_step`
    """

    def __init__(self, xml_file=None, projection_path=None, dtype=np.uint16, normalise=False, 
                 fliplr=False, mode=1):
        self.xml_file        = xml_file
        self.projection_path = projection_path
        self.dtype           = dtype
        self.normalise       = normalise
        self.fliplr          = fliplr
        self.geometry        = None
        self.shape           = None
        self.mode            = mode
        self.imaging_type    = None        
        if xml_file is not None:
            self.set_up()

    def set_up(self):
        """Parses the XML file and extracts relevant geometry information."""
        if not os.path.isfile(self.xml_file):
            raise FileNotFoundError("XML file not found.")

        tree = ET.parse(self.xml_file)
        root = tree.getroot()

        geom = root.find("Geometrie")
        recon = root.find("Recon")

        self.source_to_detector = float(geom.find("SourceDetectorDist").text)
        self.source_to_object   = float(geom.find("SourceObjectDist").text)
        self.object_to_detector = float(geom.find("ObjectDetectorDist").text)

        self.pixel_size_h       = float(recon.find("ProjectionPixelSizeX").text)
        self.pixel_size_v       = float(recon.find("ProjectionPixelSizeY").text)

        self.num_projections    = int(recon.find("ProjectionCount").text)
        self.num_pixels_h       = int(recon.find("ProjectionDimX").text)
        self.num_pixels_v       = int(recon.find("ProjectionDimY").text)

        # Load all available RAW files
        raw_files = sorted(glob.glob(os.path.join(self.projection_path, "*.raw")))
        if len(raw_files) == 0:
            raise FileNotFoundError("No RAW files found in the projection path.")

        # Always exclude the last file (1251 → 1250)
        if len(raw_files) > 1:
            raw_files = raw_files[:-1]

        self.total_available_projections = len(raw_files)  # Should be 1250

        # **Mode 1: Select 360 evenly spaced projections**
        if self.mode == 1:
            self.num_projections_to_load = min(360, self.total_available_projections)
            selected_indices = np.linspace(0, self.total_available_projections - 1, self.num_projections_to_load, dtype=int)
            self.raw_files = [raw_files[i] for i in selected_indices]
            print(f" Mode 1: Loading {self.num_projections_to_load} evenly spaced projections from {self.total_available_projections}.")

        # **Mode 2: Load all projections**
        elif self.mode == 2:
            self.num_projections_to_load = self.total_available_projections
            self.raw_files = raw_files
            print(f" Mode 2: Loading all {self.num_projections_to_load} projections.")

        # **Mode 3: Load only the central slices**
        elif self.mode == 3:
            self.num_projections_to_load = self.total_available_projections
            self.raw_files = raw_files
            print(f" Mode 3: Loading central slices from all {self.num_projections_to_load} projections.")

        self.shape = (self.num_projections_to_load, self.num_pixels_v, self.num_pixels_h)

        # Set up geometry
        self._setup_geometry()

    def _setup_geometry(self):
        """Creates the AcquisitionGeometry object based on parsed data."""
        if self.mode == 3:
            print("Mode 3: Using 2D Cone Beam Geometry.")
            self.imaging_type = "2D Cone Beam - FBP Recon"  # Set the geometry type correctly
            self.geometry = AcquisitionGeometry.create_Cone2D(
                source_position=[0, -self.source_to_object],
                detector_position=[0, self.source_to_detector - self.source_to_object]
            )
            self.geometry.set_panel(self.num_pixels_h, pixel_size=self.pixel_size_h, origin='top-left')
            self.geometry.set_labels(['angle', 'horizontal'])
        else:
            print("Mode 1 or 2: Using 3D Cone Beam Geometry.")
            self.imaging_type = "3D Cone Beam - FDK Recon"  # Set the geometry type correctly
            self.geometry = AcquisitionGeometry.create_Cone3D(
                source_position=[0, -self.source_to_object, 0],
                detector_position=[0, self.source_to_detector - self.source_to_object, 0]
            )
            self.geometry.set_panel((self.num_pixels_h, self.num_pixels_v),
                                    pixel_size=(self.pixel_size_h, self.pixel_size_v), origin='top-left')
            self.geometry.set_labels(['angle', 'vertical', 'horizontal'])

        if self.mode == 1:
            self.angles = np.linspace(-90, 270, self.num_projections_to_load, endpoint=False)
        else:
            self.angles = np.linspace(-90, 270, self.total_available_projections, endpoint=False)

        self.geometry.set_angles(self.angles, angle_unit='degree')

    def get_geometry(self):
        """Returns the AcquisitionGeometry object."""
        return self.geometry

    def get_imaging_type(self):
        """Returns the type of geometry (2D or 3D Cone beam)."""
        return self.imaging_type

    def get_mode_description(self):

        """Returns a detailed description of the selected mode."""
        if self.mode == 1:
            return f"Mode 1: 360 evenly spaced projections (out of {self.total_available_projections})"

        elif self.mode == 2:
            return f"Mode 2: Full dataset with {self.total_available_projections} projections"

        elif self.mode == 3:
            return f"Mode 3: Central slice mode (2D) from all {self.total_available_projections} projections"

        else:
            return f"Unknown mode: {self.mode}"

    def read_raw_projections(self):
        """Reads RAW projection data based on mode, applies padding & binning if needed."""
        if self.projection_path is None:
            raise ValueError("Projection data path is not set.")
        if self.shape is None:
            raise ValueError("Could not determine projection shape from XML.")

        if self.mode == 3:
            data = self._read_central_slices()
        else:
            data = self._read_full_projections()

        return data
        

    def _read_full_projections(self):
        """Loads full projection data for Mode 1 and 2."""
        num_projections, height, width = self.shape
        
        #projection_data = np.zeros((num_projections, height, width), dtype=self.dtype)

        #for i, raw_file in enumerate(self.raw_files):
        #    with open(raw_file, "rb") as f:
        #        projection_data[i] = np.fromfile(f, dtype=self.dtype).reshape((height, width))
                
        #projection_data = Parallel(n_jobs=-1, backend='loky')(
        #delayed(_read_single_raw_file)(raw_file, height, width, self.dtype)
        #for raw_file in self.raw_files )     
        
        n_threads = max(1, multiprocessing.cpu_count() // 2)

        projection_data = Parallel(n_jobs=n_threads, backend='threading')(
        delayed(_read_single_raw_file)(raw_file, height, width, self.dtype)
        for raw_file in self.raw_files )
                       
        projection_data = np.stack(projection_data, axis=0)

        return AcquisitionData(array=projection_data, geometry=self.geometry)

    def _read_central_slices(self):
        """Reads only the central slice from each RAW projection (Mode 3)."""
        central_index = self.shape[1] // 2  

        central_slices = np.zeros((self.num_projections_to_load, self.shape[2]), dtype=self.dtype)

        for i, raw_file in enumerate(self.raw_files):
            with open(raw_file, "rb") as f:
                f.seek(central_index * self.shape[2] * np.dtype(self.dtype).itemsize)
                central_slices[i] = np.fromfile(f, dtype=self.dtype, count=self.shape[2])

        return AcquisitionData(array=central_slices, geometry=self.geometry)


# In[5]:


def _read_single_raw_file(raw_file, height, width, dtype):
    with open(raw_file, "rb") as f:
        return np.fromfile(f, dtype=dtype).reshape((height, width))

def sinogram_processing(data, mode, padding_ratio=0.25, fixed_background=False, background_value=60000, 
                        apply_binning=False, binning_step=2):
    """
    Preprocess sinogram data with normalization, binning, padding, and cleaning.

    Parameters:
    - data (AcquisitionData): Raw data to preprocess.
    - padding_ratio (float): The ratio of padding to apply (default: 0.25).
    - fixed_background (bool): Whether to use a fixed background intensity value.
    - background_value (int or float): The fixed background intensity value to use if enabled.
    - apply_binning (bool): Whether to apply binning or not.
    - binning_step (int): The binning factor to apply.

    Returns:
    - data (AcquisitionData): Preprocessed data.
    """
    print(" Starting sinogram preprocessing...")

    try:
        # 🌀 Convert data to float32 before processing
        if data.array.dtype != np.float32:
            #print("🔄 Converting data to float32...")
            data.array = data.array.astype(np.float32)

        # 🌟 Normalization (Transmission to Absorption Conversion)
        '''Converting data from transmission to attenuation values (it is known to be already centered and flat field corrected)'''
        if fixed_background:
            background = background_value
            #print(f"🌟 Using fixed background intensity: {background}")
        else:
            if data.ndim == 3:
                background = data.get_slice(vertical=10, force=True).as_array().mean()
            else:
                background = data.as_array()[:, :50].mean()
            #print(f"🌟 Calculated background intensity: {background}")

        # Normalize data
        #print(" Normalizing data)
        data_array = data.as_array().astype(np.float32)  # Convert data to float32 explicitly
        data_array /= background
        data.fill(data_array)  # Update data with normalized values

        # ⚙Ensure data is explicitly float32 before conversion
        data = AcquisitionData(array=data.as_array().astype(np.float32), geometry=data.geometry)
        
        del data_array  # Clean up memory
        gc.collect()  # Force garbage collection to free memory
        gc.collect()

        data = TransmissionAbsorptionConverter(min_intensity=1e-6)(data)
        #print(" Normalization and transmission to absorption conversion done.")

        # Binning (Optional)

        '''To load a binned subset of the data, change the binning_range. 
           Here we use same binning for the horizontal and vertical dimensions which results in a the same aspect ratio'''
        if apply_binning:
            #print(f" Applying binning with factor {binning_step}...")
            #print("  Data type before Binning:", type(data))  
            #print("  Data shape before Binning:", data.shape)  
            #print("  Data dtype before Binning:", data.array.dtype) 
            if mode == 3:
                # Only vertical binning (1D sinogram)
                data = Binner(roi={'horizontal': (None, None, binning_step)})(data)  
            else:   
                # Both dimension binning (2D sinogram)            
                data = Binner(roi={'horizontal': (None, None, binning_step), 'vertical': (None, None, binning_step)})(data)
                
           # print(" Binning successful!")
            #print("  Data type after Binning:", type(data))  
            #print("  Data shape after Binning:", data.shape)  
            #print("  Data dtype after Binning:", data.array.dtype)

        #  Negative Value Clipping (Cleaning
        '''Cleaning the bad pixel from sinogram (assing 0 below 0 values)'''
        #print("🧼 Cleaning negative values...")
        data_array = data.as_array()
        data_array[data_array < 0] = 0
        data.fill(data_array)
        del data_array  # Clean up memory
        gc.collect()  # Force garbage collection to free memory
        gc.collect()
        #print(" Negative valuesin sinogram clipped to zero.")

        #  Padding (Mandatory)
        #print(" Applying padding...")
        padsize = int(data.shape[1] * padding_ratio)
        #print(f"🔹 Padding size: {padsize} pixels")
        data = Padder.edge(pad_width={'horizontal': padsize})(data)
        #print(" Padding successful!")
        #print("Data Shape after Padding:", data.shape)

        print(" Sinogram preprocessing completed.")

        return data

    except Exception as e:
        print(f" Preprocessing error: {str(e)}")
        gc.collect()
        raise


def get_def_CoR(xml_path):
    """
    Extracts default Center of Rotation from a seXML file.
    If not found, defaults to 0.0.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        recon = root.find("Recon")
        cor_element = recon.find("CenterOfRotation")
        if cor_element is not None:
            cor = float(cor_element.text)
            return cor
        else:
            save_log(" 'CenterOfRotation' not found in XML. Defaulting to 0.0")
            return 0.0
    except Exception as e:
        save_log(f" XML parsing failed: {str(e)}. Using default CoR = 0.0")
        return 0.0



# In[6]:

def  main():


    # ### Define paths
    # for projections and xml file 

    # In[18]:
    start_time = time.time()

    print("Diondo Reconstruction Setup")
    from tkinter import Tk, Label, Button, IntVar, Radiobutton, filedialog, simpledialog, messagebox

    def get_user_inputs():
        def submit():
            nonlocal xml_file, proj_folder, mode, fixed_background, background_value, binning_value
            mode = mode_var.get()
            binning_value = binning_var.get()
            fixed_background = True if bg_var.get() == 1 else False
            background_value = 60000 if fixed_background else None

            if mode == 0:
                messagebox.showerror("Error", "Please select a reconstruction mode.")
                return

            root.quit()

        root = Tk()
        header_font = ("Arial", 18, "bold")
        section_font = ("Arial", 14)
        #option_font = ("Arial", 14)
        
        root.title("Reconstruction Settings")
        # Exit warning if any selection is made
        def on_close():
            if (xml_file or proj_folder or mode_var.get() != 0 or
                bg_var.get() in [0,1] or binning_var.get() != 0 or
                save_sinogram_var.get() or save_central_slice_var.get() or save_raw_var.get()):
            
                if messagebox.askyesno("Exit", "You have made selections. Are you sure you want to exit without continuing?"):
                    save_log("User closed the reconstruction tool manually.")
                    root.destroy()
                    os._exit(0)  #  will close (PyInstaller)
            else:
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)

        # Select XML file
        Label(root, text="1. Select XML File:", font=header_font).pack(anchor='w')
        def choose_xml():
            nonlocal xml_file
            xml_file = filedialog.askopenfilename(title="Select XML file", filetypes=[("XML files", "*.xml")])
            xml_label.config(text=xml_file if xml_file else "No file selected")
        Button(root, text="Browse XML", command=choose_xml, font=section_font).pack(anchor='w')
        xml_label = Label(root, text="No file selected", font=section_font)
        xml_label.pack(anchor='w')

        # Select Projection folder
        Label(root, text="2. Select Projection Folder:", font=header_font).pack(anchor='w')
        def choose_folder():
            nonlocal proj_folder
            proj_folder = filedialog.askdirectory(title="Select the Projection Folder")
            folder_label.config(text=proj_folder if proj_folder else "No folder selected")
        Button(root, text="Browse Folder", command=choose_folder, font=section_font).pack(anchor='w')
        folder_label = Label(root, text="No folder selected", font=section_font)
        folder_label.pack(anchor='w')

        # Mode selection
        Label(root, text="3. Select Reconstruction Mode:", font=header_font).pack(anchor='w')
        mode_var = IntVar(value=1)
        modes = [
            ("1 = Load 360 evenly spaced projections (3D)", 1),
            ("2 = Load full projection set (3D)", 2),
            ("3 = Load central slice only (2D)", 3)
        ]
        for text, val in modes:
            Radiobutton(root, text=text, variable=mode_var, value=val, font=section_font).pack(anchor='w')

        # Background option (Yes / No toggle)
        Label(root, text="4. Background Option:", font=header_font).pack(anchor='w')
        bg_var = IntVar(value=1)
        Radiobutton(root, text="Yes (Use fixed value = 60000)", variable=bg_var, value=1, font=section_font).pack(anchor='w')
        Radiobutton(root, text="No (Auto calculate from data)", variable=bg_var, value=0, font=section_font).pack(anchor='w')

        # Binning option
        Label(root, text="5. Binning Option:", font=header_font).pack(anchor='w')
        binning_var = IntVar(value=1)   
        binning_options = [
            ("No binning", 0),
            ("Binning = 2", 2),
            ("Binning = 4", 4),
            ("Binning = 8", 8)
        ]
        for text, val in binning_options:
            Radiobutton(root, text=text, variable=binning_var, value=val, font=section_font).pack(anchor='w')
            
            
        # Geometry save option (Yes / No)
        Label(root, text="6. What do you want to save?", font=header_font).pack(anchor='w')
        save_sinogram_var = IntVar(value=1)
        #save_geometry_var = IntVar(value=1)
        save_central_slice_var = IntVar(value=1)
        save_raw_var = IntVar(value=1)

        Checkbutton(root, text="Sample Sinogram (TIFF)", variable=save_sinogram_var, font=section_font).pack(anchor='w')
        #Checkbutton(root, text="Geometry Setup Image (PNG)", variable=save_geometry_var).pack(anchor='w')
        Checkbutton(root, text="Reconstructed Central Slice (TIFF)", variable=save_central_slice_var, font=section_font).pack(anchor='w')
        Checkbutton(root, text="Reconstructed Volume as RAW", variable=save_raw_var, font=section_font).pack(anchor='w')

        # Submit
        Button(root, text="Submit", command=submit, font=header_font).pack(pady=10)

        # Initial values
        xml_file = None
        proj_folder = None
        mode = None
        fixed_background = None
        background_value = None
        binning_value = None

        root.mainloop()
        root.destroy()

        if not xml_file:
            raise ValueError("No XML file selected.")
        if not proj_folder:
            raise ValueError("No projection folder selected.")

        return (xml_file, proj_folder, mode, fixed_background, background_value,
            binning_value, save_sinogram_var.get(), 
            save_central_slice_var.get(), save_raw_var.get())



    (xml_file_name, proj_file_name, mode, fixed_background, background_value,
    binning_value, save_sinogram,  save_central_slice, save_raw_file) = get_user_inputs()

     
    path = os.path.dirname(xml_file_name)

    # Define save path
    save_base_path = path 
    # Extract dataset name from the path
    dataset_name = os.path.basename(path)  # Get the last folder name
    dataset_name = dataset_name.replace(" ", "_")  # Remove spaces

    # In[20]:


    save_log("Data uploading processing started.")
    uploading_start_time = time.time()
    data_reader = DiondoDataReader(xml_file = xml_file_name, projection_path = proj_file_name, mode = mode)
    data = data_reader.read_raw_projections()
    uploading_end_time = time.time()
    uploading_duration = uploading_end_time - uploading_start_time
    save_log(f"Loaded projections from path: {proj_file_name}")
    save_log(f"Selected mode: {data_reader.get_mode_description()}")
    save_log(f"Number of loaded projections: {data_reader.num_projections_to_load}")
    save_log(f"Imaging type: {data_reader.get_imaging_type()}")
    save_log(f"Data uploading completed in {uploading_duration:.2f} seconds.")


    # In[21]:


    save_log("Sinogram processing started.")
    sinogram_start_time = time.time()

    padding_size = int(data.shape[1] * 0.25)

    apply_binning = binning_value not in [None, 0]

    data = sinogram_processing(
        data,
        mode=mode,
        padding_ratio=0.25,
        fixed_background=fixed_background,
        background_value=background_value,
        apply_binning=apply_binning,
        binning_step=binning_value if apply_binning else None
    )

    sinogram_end_time = time.time()
    sinogram_duration = sinogram_end_time - sinogram_start_time

    binning_status = "Yes" if apply_binning else "No"

    save_log(f"Binning applied: {binning_status}, Binning factor: {binning_value} ")
    save_log(f"Padding applied: Yes, Padding size: {padding_size} pixels")
    save_log(f"Sinogram processing completed in {sinogram_duration:.2f} seconds.")


    # In[22]:



    if save_sinogram:
        fig = plt.figure()
        if mode == 3:
            plt.imshow(data.as_array(), cmap="gray")
            plt.title("Sinogram")
            #plt.show(block=False)
        else:
            sinogram_array = data.as_array()
            center_index = sinogram_array.shape[1] // 2
            middle_slice = sinogram_array[:, center_index, :]
            plt.imshow(middle_slice, cmap="gray")
            plt.title("Sample Sinogram")
            #plt.show(block=False)

        plt.colorbar()
        plt.xlabel("Detector Pixels")
        plt.ylabel("Angle")

        sinogram_file_name = f"sinogram_{dataset_name}.tiff"
        sinogram_file_path = os.path.join(path, sinogram_file_name)
        plt.savefig(sinogram_file_path, dpi=300)
        plt.close(fig)
        save_log(f"Sample sinogram saved at: {sinogram_file_path}")


    # View the geometry with show_geometry to display information about the source and detector setup

    # In[23]:


    geometry = data_reader.get_geometry()
    #show_geometry(geometry)
    print(geometry)
    save_log("=== Geometry Info ===")
    save_log(str(geometry))  # Save to the log file 


    # To set up the FDK algorithm we must specify the size/geometry of the reconstruction volume. Here we use the default one:

    # In[24]:


    ig   = data_reader.geometry.get_ImageGeometry()
    #print(ig)
    save_log("=== Image Info ===")
    save_log(str(ig))  # Save to the log file 


    # Here we use a CIL Processor to determine the offset and update the geometry. It works by doing FBP/FDK reconstructions for a range of offset parameters, evaluates a quality metric based on image sharpness and searches for the best offset. 

    # In[ ]:

    print("Running Centre of Rotation Correction...")
    save_log(f"Centre of Rotation Correction started.")
    corc_start_time   = time.time()
    correction_success = False  # Status flag
    try:
    # Step 1: Try with default parameters
        data = CentreOfRotationCorrector.image_sharpness()(data)
        correction_result = data.geometry.get_centre_of_rotation(distance_units='pixels')
        save_log("CoR correction successful with default parameters.")
        correction_success = True  

    except Exception as e:
        save_log("CoR correction failed with default parameters. Retrying with relaxed parameters...")
        print("CoR correction failed with default parameters. Retrying with relaxed parameters...")

        try:
        # Step 2: Retry with wider search range and relaxed tolerance
            s_range = data.shape[2] // 3  # Use one-third of horizontal dimension
            tol = 1 / 100  # Relax tolerance (default is 1/200)
            data = CentreOfRotationCorrector.image_sharpness(search_range=s_range, tolerance=tol)(data)
            correction_result = data.geometry.get_centre_of_rotation(distance_units='pixels')
            save_log(" CoR correction successful with relaxed parameters.")
            correction_success = True 

        except Exception as e:
        # Step 3: Fallback – use default value from the XML file
            save_log("CoR correction failed again. Reverting to default value from XML.")
            print("CoR correction failed again. Reverting to default value from XML.")
            def_COR = get_def_CoR(xml_file_name)
            offset_value = def_COR
            offset_unit = "pixels"
            angle_value = 0.0
            angle_unit = "degrees"
            save_log(f" Using default CoR value from XML: {def_COR} pixels")

        # Manually update geometry with fallback CoR value
            updated_geometry = data.geometry.copy()
            updated_geometry.config.system.rotation_axis.position[0] = offset_value
            data.geometry = updated_geometry
            #correction_success = True


    # Step 4: If any correction was successful, extract values from result
    if correction_success:
        offset_value = correction_result['offset'][0]
        offset_unit = correction_result['offset'][1]
        angle_value = correction_result['angle'][0]
        angle_unit = correction_result['angle'][1]

    end_corc_time = time.time()
    corc_duration = end_corc_time - corc_start_time

    # saving log file
    save_log(f"Offset: {offset_value:.4f} image {offset_unit}, Angle: {angle_value:.4f} {angle_unit}")
    #save_log(f"Offset: {offset_value:.4f} voxels, Angle: {angle_value:.4f} {angle_unit}")
    save_log(f"Centre of Rotation Correction completed in {corc_duration:.2f} seconds.")
    print("Centre of Rotation Correction Completed")

    # We can then create the FBP/FDK algorithm and reconstruct the data depending on the seleccted mode:

    # In[ ]:

    save_log(f"Reconstruction started.")
    # **Select the appropriate reconstruction method based on the mode**
    recon_start_time = time.time()

    print("Running Reconstruction")

    if data_reader.mode in [1, 2]:  
        print("Running FDK Reconstruction (3D Mode)")
        fdk_recon = FDK(data, ig).run()
        recon_image = fdk_recon  # Store the 3D reconstruction result
    else:  
        print("Running FBP Reconstruction (2D Mode)")
        fbp_recon = FBP(ig, data.geometry)(data)
        recon_image = fbp_recon  # Store the 2D reconstruction result

    end_recon_time = time.time()
    total_recon_duration = end_recon_time - recon_start_time

    recon_shape = recon_image.shape

    save_log(f"Reconstruction completed. Output image size: {recon_shape}")
    save_log(f"Total reconstruction time: {total_recon_duration:.2f} seconds.")
    print("Reconstruction is completed")

    # In[ ]:


    # **Visualization**
    if mode == 3:  # If 2D reconstruction was performed
        plt.imshow(recon_image.as_array(), cmap="gray")
        plt.colorbar()
        plt.title("Central Slice (FBP - 2D)")
        plt.show(block=False)
    else:  # If 3D reconstruction was performed
        central_slice = recon_image.shape[0] // 2  # **Select the central slice**
        plt.imshow(recon_image.as_array()[central_slice, :, :], cmap="gray")
        plt.colorbar()
        plt.title("Central Slice (FDK - 3D)")
        plt.show(block=False)



    # ### Saving the reconstructed file
    # 
    # We can save the central slice as a tiff file

    # In[ ]:
    # Number of projections used
    P = data.shape[0]  # This ensures P correctly reflects the number of loaded projections


   # TIFF central slice save
    if save_central_slice:
        if data_reader.mode == 3:
            # 2D case
            Y, X = recon_image.shape
            file_name = f"FBP_recon_{dataset_name}_{P}proj_CSR_{Y}x{X}_32bit.tiff"
            save_path = os.path.join(save_base_path, file_name)
            tiff.imwrite(save_path, recon_image.as_array().astype(np.float32))
            print(f"Central slice (2D) saved at: {save_path}")
            save_log(f"Central slice saved at: {save_path}")
        else:
            # 3D case
            X, Y, Z = recon_image.shape
            file_name = f"FDK_recon_{dataset_name}_{P}proj_{Y}x{Z}_central_slice_32bit.tiff"
            save_path = os.path.join(save_base_path, file_name)
            middle = recon_image.as_array()[recon_image.shape[0] // 2, :, :]
            tiff.imwrite(save_path, middle.astype(np.float32))
            print(f"Central slice (3D) saved at: {save_path}")
            save_log(f"Central slice saved at: {save_path}")


    # In[ ]:


    # Record end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    print(f"Total time: {elapsed_time:.2f} s")
    save_log(f"Total time: {elapsed_time:.2f} seconds.")


    # ### Raw File (2D-3D)
    # Saving the raw file

    # In[ ]:

    if save_raw_file:
        if data_reader.mode == 3:
            Y, X = recon_image.shape
            file_name = f"FBP_recon_{dataset_name}_{P}proj_CSR_{Y}x{X}_32bit.raw"
            save_path = os.path.join(save_base_path, file_name)
            recon_array = recon_image.as_array().astype(np.float32)
            with open(save_path, 'wb') as f:
                recon_array.tofile(f)
            print(f"2D RAW saved: {save_path}")
            save_log(f"2D RAW saved: {save_path}")
        else:
            Z, Y, X = recon_image.shape
            file_name = f"FDK_recon_{dataset_name}_{P}proj_{Y}x{X}x{Z}_32bit.raw"
            save_path = os.path.join(save_base_path, file_name)
            writer = RAWFileWriter(data=recon_image, file_name=save_path)
            writer.write()
            print(f"3D RAW saved: {save_path}")
            save_log(f"3D RAW saved: {save_path}")


    # In[ ]:


    # Generate log file name dynamically
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{dataset_name}_reconstruction_{timestamp}.txt"

    # Define the log directory and file path
    log_base_path = os.path.join(path, "logs")
    os.makedirs(log_base_path, exist_ok=True)
    log_file_path = os.path.join(log_base_path, log_file_name)

    # Now, save the collected logs
    save_log("All reconstruction processes completed successfully.", log_file_path)
    # Start the logging process
    print("Showing geometry setup (user must close it manually)...")
    # === Geometry visualization: at the very end ===
    show_geometry(geometry)
    


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
