// File:    inputtensor.cpp
//
// Purpose: computes the input tensor for a photo with the CImg and
//			frugally-deep package. JPG images, due to their lossy nature,
//			are rendered differently on Unix or Windows. The output
//		 	of this program can be used as input for tpredict.exe.
//          In this way one can find out if different prediction
//			results are due to differences in JPG rendering of problems
//			with the model.
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

#if defined _MSC_VER
#define cimg_OS 2
#endif

#define cimg_use_png
#define cimg_use_jpeg
#include "CImg.h"

#ifdef Success
  #undef Success
#endif

#include <fdeep/fdeep.hpp>

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

fdeep::tensor cimg_to_tensor(const cimg_library::CImg<unsigned char>& image, float mean[])
{
    const int width = image.width();
    const int height = image.height();
    const int channels = image.spectrum();

    std::vector<float> pixels;
    pixels.reserve(height * width * channels);

    // CImg stores the pixels of an image non-interleaved:
    // http://cimg.eu/reference/group__cimg__storage.html
    // This loop changes the order to interleaved,
    // e.e. RRRGGGBBB to RGBRGBRGB for 3-channel images.
	
	// png rgba files do have four channels, we only need the first three !
    for (int y = 0; y < height; y++) {
       for (int x = 0; x < width; x++) {
          for (int c = 0; c < 3; c++) {
             pixels.push_back(image(x, y, 0, c)-mean[c]);
          }
       }
    }

    return tensor_from_floats(pixels.data(), height, width, 3, 0.0f, 255.0f);
}

inline std::string show_tensor(const fdeep::tensor& t) {
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
    cimg_library::CImg<unsigned char> img;
	string myname = argv[0];
	
    // In Windows argv[0] is an absolute path name, so determine last filename
    size_t slash_or_backslash = myname.rfind("\\");
    if (slash_or_backslash != string::npos) {
        myname = myname.substr(slash_or_backslash+1);
    }

	if (argc != 3) {
       cerr << "Usage: " << myname << " cfgfile image" << endl; 
	   cerr << "\nComputes input tensor from resized image\n";
	   cerr << "You can use the saved output for prediction with tpredict.exe\n\n";
       exit(1);
    }

    string cfgfn = argv[1];
	// Get the extension and check if this is a text file
    string extension = getFileExtension(cfgfn);				
	if (strcmp(extension.c_str(),".cfg") != 0) {
        cerr << "File " << cfgfn << " no .cfg file\n\n";
		return(1);
	}
	
	// load configuration file
	string cfgfilename(cfgfn);

    // Default values used in learning cats & dogs model Jason Brownlee
	int WHC[3] = {224, 224, 3};
	float mean[3] = {123.68f, 116.779f, 103.939f}; // mean RGB values for VGG16 prelearned model
	std::string categories = "{Cats = 0, Dogs = 1}";
	
    // Check if the cfg file exist; otherwise use default values
	if (exists(cfgfilename)) {
		ConfigFile cfg(cfgfilename);
		
		if (cfg.keyExists("shapeWHC")) {
		   std::string shapeWHC = cfg.getValueOfKey<std::string>("shapeWHC", "Unknown");
		   stringstream ss(shapeWHC);
		   WHC[0] = nextnumber(ss);
		   WHC[1] = nextnumber(ss);
		   WHC[2] = nextnumber(ss);
		   cerr << "Shape from cfg file = {" << WHC[0] << "," << WHC[1] << "," << WHC[2] << "}" << endl;
		}
	   
		if (cfg.keyExists("meanRGB")) {
		   std::string meanRGB = cfg.getValueOfKey<std::string>("meanRGB");
		   stringstream ss(meanRGB);
		   mean[0] = nextnumber(ss);
		   mean[1] = nextnumber(ss);
		   mean[2] = nextnumber(ss);
		   cerr << "MeanRGB from cfg file = {" << mean[0] << ", " << mean[1] << ", " << mean[2] << "}" << endl;
		}

		if (cfg.keyExists("categories")) {
		   categories = cfg.getValueOfKey<std::string>("categories");
		}
	}
		
	img = cimg_library::CImg<unsigned char>(argv[2]);
	if (img.spectrum() == 1) {
       cerr << "We can only handle images with three channels\n\n";
       exit(1);
    }
	if (img.width() != WHC[0] || img.height() != WHC[1]) img.resize(WHC[0], WHC[1]);

	// Use the correct scaling, i.e. low and high; subtract mean values
    const auto input = cimg_to_tensor(img, mean);
		
	// print debugging info
	string inputstr = show_tensor(input);

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
	
    return 0;
}