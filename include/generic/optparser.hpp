// ============================================================================
// Copyright (C), QUEST.
// All rights reserved. See files LICENSE for details.
//
// File: optparser.hpp
// Author: Liang Pan
// E-mail: lightpl@stu.xmu.edu.cn
// ============================================================================

#ifndef ENSIGN_OPTPARSER
#define ENSIGN_OPTPARSER

#include <vector>
#include <string>
#include <iostream>

#include <generic/storage.hpp>

namespace Ensign
{

/** Class for parsing command-line options.

    Construct the parser with argc and argv and register options with
    AddOption. Pass nullptr as the short name for an option with only a long name.
    Supported types are bool, int, float, double, const char*, and std::string.
*/
class OptionsParser
{
public:
   enum OptionType { INT, FLOAT, DOUBLE, STRING, STD_STRING, VECTOR, ENABLE, DISABLE};

private:
   struct Option
   {
      OptionType type;
      void *var_ptr;
      const char *short_name;
      const char *long_name;
      const char *description;
      bool required;

      // 使用显式空构造，遵循项目“禁止 default 构造”约定。
      Option()
      {
         type = INT;
         var_ptr = nullptr;
         short_name = nullptr;
         long_name = nullptr;
         description = nullptr;
         required = false;
      }

      Option(OptionType type_, void *var_ptr_, const char *short_name_,
             const char *long_name_, const char *description_, bool req)
         : type(type_), var_ptr(var_ptr_), short_name(short_name_),
           long_name(long_name_), description(description_), required(req) { }
   };

   int argc;
   char **argv;
   std::vector<Option> options;
   std::vector<int> option_check;
   // error_type can be:
   //  0 - no error
   //  1 - print help message
   //  2 - unrecognized option at argv[error_idx]
   //  3 - missing argument for the last option argv[argc-1]
   //  4 - option with index error_idx is specified multiple times
   //  5 - invalid argument in argv[error_idx] for option in argv[error_idx-1]
   //  6 - required option with index error_idx is missing
   int error_type, error_idx;

   static void WriteValue(const Option &opt, std::ostream &out);

public:

   /// Construct a command line option parser with 'argc_' and 'argv_'.
   OptionsParser(int argc_, char *argv_[])
      : argc(argc_), argv(argv_)
   {
      error_type = error_idx = 0;
   }

   /** @brief Add a boolean option and set 'var' to receive the value.
       Enable/disable tags are used to set the bool to true/false
       respectively. */
   void AddOption(bool *var, const char *enable_short_name,
                  const char *enable_long_name, const char *disable_short_name,
                  const char *disable_long_name, const char *description,
                  bool required = false)
   {
      options.emplace_back(Option(ENABLE, var, enable_short_name, enable_long_name,
                            description, required));
      options.emplace_back(Option(DISABLE, var, disable_short_name, disable_long_name,
                            description, required));
   }

   /// Add an integer option and set 'var' to receive the value.
   void AddOption(int *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.emplace_back(Option(INT, var, short_name, long_name, description,
                            required));
   }

   /// Add a double option and set 'var' to receive the value.
   void AddOption(float *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.emplace_back(Option(FLOAT, var, short_name, long_name, description,
                            required));
   }

   /// Add a double option and set 'var' to receive the value.
   void AddOption(double *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.emplace_back(Option(DOUBLE, var, short_name, long_name, description,
                            required));
   }

   /// Add a string (char*) option and set 'var' to receive the value.
   void AddOption(const char **var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.emplace_back(Option(STRING, var, short_name, long_name, description,
                            required));
   }

   /// Add a string (std::string) option and set 'var' to receive the value.
   void AddOption(std::string *var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.emplace_back(Option(STD_STRING, var, short_name, long_name, description,
                            required));
   }

   void AddOption(multi_array<double,1> *var, const char *short_name,
                const char *long_name, const char *description,
                bool required = false)
  {
    options.emplace_back(Option(VECTOR, var, short_name, long_name, description,
                          required));
  }

   /** @brief Parse the command-line options.
       Note that this function expects all the options provided through the
       command line to have a corresponding AddOption. In particular, this
       function cannot be used for partial parsing. */
   void Parse();

   /// Parse the command line and print the selected options to the given stream.
   /// A help request exits successfully. Invalid input exits with an error.
   void ParseCheck(std::ostream &out = std::cout);

   /// Return true if the command line options were parsed successfully.
   bool Good() const { return (error_type == 0); }

   /// Return true if we are flagged to print the help message.
   bool Help() const { return (error_type == 1); }

   /// Print the options
   void PrintOptions(std::ostream &out) const;

   /// Print the error message
   void PrintError(std::ostream &out) const;

   /// Print the help message
   void PrintHelp(std::ostream &out) const;

   /// Print the usage message
   void PrintUsage(std::ostream &out) const;
};

}

#endif // ENSIGN_OPTPARSER
