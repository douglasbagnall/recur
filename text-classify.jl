using LightXML
using ArgParse
using StrPack

#these need to match the rnn_char_flags enum in charmodel.h
const CASE_INSENSITIVE = uint32(1)
const UTF8             = uint32(2)
const COLLAPSE_SPACE   = uint32(4)

type LangBlock
    lang::String
    text::String
end

const NO_LANG = "xx"
const NO_CLASS = 0xFF
#const XMLNAME = "/home/douglas/corpora/maori-legal-papers/GorLaws.xml"
const XMLNAME = "/home/douglas/corpora/maori-legal-papers/Gov1909Acts.xml"
const DEFAULT_LAG = 10

const COLOURS = ["normal" => "\033[00m",
                 "dark_red" => "\033[00;31m",
                 "red" =>"\033[01;31m",
                 "dark_green" => "\033[00;32m",
                 "green" => "\033[01;32m",
                 "yellow" => "\033[01;33m",
                 "dark_yellow" => "\033[00;33m",
                 "dark_blue" => "\033[00;34m",
                 "blue" => "\033[01;34m",
                 "purple" => "\033[00;35m",
                 "magenta" => "\033[01;35m",
                 "dark_cyan" => "\033[00;36m",
                 "cyan" => "\033[01;36m",
                 "grey" => "\033[00;37m",
                 "white" => "\033[01;37m"]


function clean_text(t, flags)
    if flags & COLLAPSE_SPACE != 0
        t = replace(t, r"\s+"s, " ")
    end
    if flags & CASE_INSENSITIVE != 0
        t = lowercase(t)
    end
    return t
end

function getlangstrings(el, lang_blocks, lang::String, flags::Uint32)
    tagname = name(el)
    if tagname == "foreign"
        lang = NO_LANG
    elseif has_attribute(el, "lang")
        lang = attribute(el, "lang")
    end
    for c::XMLNode in child_nodes(el)
        if is_textnode(c)
            t = clean_text(content(c), flags)
            push!(lang_blocks, LangBlock(lang, t))
        elseif is_elementnode(c)
            e = XMLElement(c)
            getlangstrings(e, lang_blocks, lang, flags)
        end
    end
end


function xml_getlangstrings(xmlname::String)
    flags::Uint32 = CASE_INSENSITIVE | UTF8 | COLLAPSE_SPACE
    xdoc = parse_file(xmlname)
    xroot = root(xdoc)
    lang_blocks_raw = LangBlock[]
    getlangstrings(xroot, lang_blocks_raw, NO_LANG, flags)

    langs = Set()
    full_text = ""
    prev_was_space = false
    lang_blocks = LangBlock[]
    for e in lang_blocks_raw
        text_is_space = e.text == " "
        if prev_was_space && text_is_space && (flags & COLLAPSE_SPACE) != 0
            continue
        end
        prev_was_space = text_is_space
        if e.lang != NO_LANG
            push!(langs, e.lang)
        end
        
        push!(lang_blocks, e)
        full_text = string(full_text, e.text)
    end
    langs2 = [NO_LANG => NO_CLASS]
    for (i, x) in enumerate(langs)
        langs2[x] = uint8(i)
    end
    alphabet = zeros(Cint, 257)
    collapse_chars = zeros(Cint, 257)
    a_len = Cint[0]
    c_len = Cint[0]
    threshold = 1e-4
    digit_adjust = 0.5
    alpha_adjust = 2.0

    ccall((:rnn_char_find_alphabet_s, "./libcharmodel.so"),
          Cint, (Ptr{Uint8}, #*text,
                 Cint,       #int len,
                 Ptr{Cint},  #int *alphabet,
                 Ptr{Cint},  #int *a_len,
                 Ptr{Cint},  #int *collapse_chars,
                 Ptr{Cint},  #int *c_len,
                 Float64,    #double threshold,
                 Float64,    #double digit_adjust,
                 Float64,    #double alpha_adjust
                 Uint32,     #u32 flags
                 ),
          full_text, length(full_text),
          alphabet, a_len,
          collapse_chars, c_len,
          threshold, digit_adjust, alpha_adjust, flags)

    labelled_text = Uint8[]
    #labelled_text = Array{LabelledChar, 1}
    i = 1
    for e in lang_blocks
        for c in e.text
            push!(labelled_text, langs2[e.lang])
            push!(labelled_text, uint8(c))
        end
    end
    @printf("%d %d, %d, %d\n", labelled_text[1], labelled_text[2],
            labelled_text[3], labelled_text[4])
    return labelled_text, langs2
end

function print_colourised_text(labelled_text)
    colours = [COLOURS[x] for x in ["red", "blue", "green", "yellow"]]
    no_class_colour = COLOURS["cyan"]
    prev_label = NO_CLASS
    label = true
    for c in labelled_text
        #println(" ", c)
        if label
            if prev_label != c
                colour = (c == NO_CLASS) ? no_class_colour : colours[c]
                print(colour)
                prev_label = c
            end
        else
            @printf("%c", c)
        end
        label = ! label
    end
    print(COLOURS["normal"])
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--file", "-f"
        help = "XML file to parse"
        default = XMLNAME

        "--colourise-text"
        action = :store_true
        help = "show the classification of input using colour"

        "--lag", "-l"
        arg_type = Int
        default = DEFAULT_LAG
        help = "Classes lag this many characters behind"
    end
    args = parse_args(s)

    labelled_text, langs2 = xml_getlangstrings(args["file"])
    @printf("%d %d, %d, %d\n", labelled_text[1], labelled_text[2],
            labelled_text[3], labelled_text[4])



    if args["lag"] != 0
        ccall((:rnn_char_adjust_text_lag, "./libcharmodel.so"), Void,
              (Ptr{Uint8}, Cint, Cint),
              labelled_text,
              length(labelled_text) / 2, args["lag"])
    end

    @printf("%d %d, %d %d\n", labelled_text[1], labelled_text[2],
            labelled_text[3], labelled_text[4]
            )

    println(langs2, args)
    if args["colourise-text"]
        print_colourised_text(labelled_text)
    end
end


main()
