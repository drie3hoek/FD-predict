// File:    tpredict.cpp
//
// Purpose: tpredict to which Genus (Archips or Clepsis) a photo of a
// 	    	leafroller belongs.
//          Comparable to the cats and dogs problem explained by
//          Jason Brownlee. This program does use the C++ frugally-deep
//			source files from Tobias Hermann. It does allow me to run
//          on a cpu that does not support Tensorflow/Keras.
//
// Author:  Piet Tutelaers
// Date:    May 2020

/* For next line see issue https://developercommunity.visualstudio.com/content/problem/93889/error-c2872-byte-ambiguous-symbol.html
*/
#define _HAS_STD_BYTE 0

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

static void show_usage(string name)
{
    cerr << "Usage: " << name << " <options> model tensorfile..." << endl << endl;
	cerr << "Model\t\tJSON model file\n\n"
         << "Options:\n"
		 << "\t-d\t--debug\t\tDebug (print) input given to model\n"
         << "\t-h\t--help\t\tShow this help message\n"
         << endl;
}

/*
* Get File extension from File path or File Name
*/
string getFileExtension(string filePath) {
   // Find the last position of '.' in given string
   size_t pos = filePath.find_last_of('.');
   // If last '.' is found
   if (pos != string::npos) {
      // return the substring
      return filePath.substr(pos);
   }
   // In case of no extension return empty string
   return "";
}

inline bool exists (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

float nextnumber(std::istream & input) {
	bool plus = true;
	int divisor = 0;
    float number = 0; 
    char c;

    // skip till start of a number
    while (input.get(c)) {
       if (isdigit(c)) { number = (float) (c - '0'); break; }
	   if (c == '-' || c == '+') {
		  if (c == '-') plus = false;
		  continue;
	   }
	}
	if (input.eof()) {
	   cout << "[EoF reached]\n";
	   exit(1);
	}
	
    // read number
    while (input.get(c)) {
		if (c == '-' || c == '+') {
			cerr << "Unexpected sign" << endl;
			exit(1);
		}
		if (c == '.') {
			divisor = 10.0;
			continue;
		}
		if (isdigit(c)) {
			if (divisor == 0) number = number * 10 + (c - '0');
			else {
				number = number + (float) (c - '0') / divisor;
				divisor = divisor * 10.0;
			}
			continue;
		}
        // character not belonging to a number
		break;
    }

    // return number
    if (plus) return number;
	else return -number;
}

#include "cfgfile.hpp"

#include <fdeep/fdeep.hpp>

// fdeep::tensor array_to_tensor(int width, int height, int channels, float data[width][height][channels])

// Converts a memory block holding 8-bit values into a tensor.
// Data must be stored row-wise (and channels_last).
// Scales the values from range [0, 255] into [low, high].
// Example:
//     With low = 0.0 and high = 1.0 every value is essentially divided by 255.
// May be used to convert an image (bgr, rgba, gray, etc.) to a tensor.
inline fdeep::tensor tensor_from_floats(const float* value_ptr,
    std::size_t height, std::size_t width, std::size_t channels,
    fdeep::internal::float_type low = 0.0f, fdeep::internal::float_type high = 1.0f)
{
    const vector<float> bytes(
        value_ptr, value_ptr + height * width * channels);
    auto values = fplus::transform_convert<fdeep::float_vec>(
        [low, high](float b) -> fdeep::internal::float_type
    {
        return fplus::reference_interval(low, high,
            static_cast<fdeep::internal::float_type>(0.0f),
            static_cast<fdeep::internal::float_type>(255.0f),
            static_cast<fdeep::internal::float_type>(b));
    }, bytes);
    return fdeep::tensor(fdeep::tensor_shape(height, width, channels), std::move(values));
}

template <int width, int height, int channels>
fdeep::tensor array_to_tensor(float (&data)[width][height][channels])
{
    std::vector<float> pixels;
    pixels.reserve(width * height * channels);

    // CImg stores the pixels of an image non-interleaved:
    // http://cimg.eu/reference/group__cimg__storage.html
    // This loop changes the order to interleaved,
    // e.e. RRRGGGBBB to RGBRGBRGB for 3-channel images.

    for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                pixels.push_back(data[x][y][c]);
            }
        }
    }

    return tensor_from_floats(pixels.data(), width, height, channels,
        0.0f, 255.0f);
}

inline std::string show_tensor(const fdeep::tensor& t)
{
    const auto xs = *t.as_vector();
    const auto test_strs = fplus::transform(
        fplus::fwd::show_float_fill_left(' ', 0, 3), xs);
    const auto max_length = fplus::size_of_cont(fplus::maximum_on(
        fplus::size_of_cont<std::string>, test_strs));
    const auto strs = fplus::transform(
        fplus::fwd::show_float_fill_left(' ', max_length, 3), xs);
    return fplus::show_cont(
        fplus::split_every(t.shape().size_dim_5_,
            fplus::split_every(t.shape().size_dim_4_,
                fplus::split_every(t.shape().height_,
                    fplus::split_every(t.shape().width_,
                        fplus::split_every(t.shape().depth_, strs))))));
}

int main(int argc, char* argv[]) {
	string myname = argv[0];
	bool debug = false;

	// In Windows argv[0] is an absolute path name, so determine last filename 
	size_t slash_or_backslash = myname.rfind("\\");
	if (slash_or_backslash != string::npos) {
		myname = myname.substr(slash_or_backslash+1);
	}
	
	string modelfn;
	
    if (argc < 3) {
        show_usage(myname);
        return 1;
    }
    vector<string> tensorfiles;

    // collect options, model filename and filenames of images
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d") || (arg == "--debug")) {
			debug = true;
		} else if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
		} else if (arg.compare(0,1,"-") == 0) {
			cerr << "invalid option " << arg << endl << endl;
            show_usage(myname);
			return(1);
        } else { // no option
			string extension = getFileExtension(arg);
            if (strcmp(extension.c_str(), ".json") == 0) {
                modelfn = argv[i];
			}
			else tensorfiles.push_back(argv[i]);
        }		
    }
	
	if (tensorfiles.size() == 0) {
		cerr << "No tensorfiles given" << endl << endl;
        show_usage(myname);
		return(1);
	}
	for (vector<string>::iterator it = tensorfiles.begin(); it != tensorfiles.end(); it++) {
        // does the file exist?
		if (! exists(*it)) {
			cerr << "File " << *it << " doesn't exist\n\n";
			return(1);
        }
		// Get the extension and check if this is a text file
        string extension = getFileExtension(*it);				
	    if (strcmp(extension.c_str(),".txt") != 0) {
            cerr << "File " << *it << " no .txt file\n\n";
			return(1);
	    }
	}

    // Check if the file exist
	if (! exists(modelfn)) {
		cerr << "File " << modelfn << " doesn't exist\n\n";
		return(1);
    }
	
	// load configuration file with the same name as modelfn but with extension .cfg
	string cfgfilename(modelfn);
	cout << cfgfilename << endl;
	size_t point = cfgfilename.find_last_of(".");
	cfgfilename.replace(point, 5, ".cfg\0");


    // Default values used in learning cats & dogs model Jason Brownlee
	int WHC[3] = {224, 224, 3};
	float mean[3] = {123.68f, 116.779f, 103.939f}; // mean RGB values for VGG16 prelearned model
	std::string categories = "{Cats = 0, Dogs = 1}";
	
    // Check if the cfg file exist
	if (exists(cfgfilename)) {
		ConfigFile cfg(cfgfilename);
		
		if (cfg.keyExists("shapeWHC")) {
		   std::string shapeWHC = cfg.getValueOfKey<std::string>("shapeWHC", "Unknown");
		   stringstream ss(shapeWHC);
		   WHC[0] = nextnumber(ss);
		   WHC[1] = nextnumber(ss);
		   WHC[2] = nextnumber(ss);
		   cout << "Shape from cfg file = {" << WHC[0] << "," << WHC[1] << "," << WHC[2] << "}" << endl;
		}
	   
		if (cfg.keyExists("meanRGB")) {
		   std::string meanRGB = cfg.getValueOfKey<std::string>("meanRGB");
		   stringstream ss(meanRGB);
		   mean[0] = nextnumber(ss);
		   mean[1] = nextnumber(ss);
		   mean[2] = nextnumber(ss);
		   cout << "MeanRGB from cfg file = {" << mean[0] << ", " << mean[1] << ", " << mean[2] << "}" << endl;
		}

		if (cfg.keyExists("categories")) {
		   categories = cfg.getValueOfKey<std::string>("categories");
		}
	}
		

    // load model
    const auto model = fdeep::load_model(modelfn);

    // predict list of tensorfiles with this model
	vector<string>::iterator it;
	int i = 0;
	for (it = tensorfiles.begin(); it != tensorfiles.end(); it++, i++) {
		const int width = 224, height = 224;
		const int channels = 3;
	
		float data[width][height][channels];
		
		ifstream input(*it);

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				for (int c = 0; c < channels; c++) {
					data[i][j][c] = nextnumber(input);
				}
			}
		}
		input.close();

		// Convert data (3D array) to tensor
        const auto inputtensor = array_to_tensor(data);

		// print debugging info
		if (debug) {
			string inputstr = fdeep::show_tensor(inputtensor);

			// replace ], by ],\n for readability
			size_t index = 0;
			while (true) {
				/* Locate the substring to replace. */
				index = inputstr.find("],", index);
				if (index == std::string::npos) break;

				/* Make the replacement. */
				inputstr.replace(index, 3, "],\n");

				/* Advance index forward so the next iteration doesn't pick it up as well. */
				index += 3;
			}
		
			cout <<  inputstr << endl;
		}

        float fresult= model.predict_class_with_confidence({inputtensor}).second;
        cout << *it << " belongs to " << setprecision(8) << fixed << fresult << endl;
    }
	cout << "Categories = " << categories << endl;

    return 0;
}
