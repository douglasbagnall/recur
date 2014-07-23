using LightXML
using ArgParse

# parse ex1.xml:
# xdoc is an instance of XMLDocument, which maintains a tree structure

type LangBlock
    lang::String
    text::String
end

type LabelledChar
    label::Uint8
    letter::Uint8
end

const NO_LANG = "xx"
const XMLNAME = "/home/douglas/corpora/maori-legal-papers/GorLaws.xml"
const DEFAULT_LAG = 10


function getlangstrings(el, lang_blocks, lang)
    tagname = name(el)
    if tagname == "foreign"
        lang = NO_LANG
    elseif has_attribute(el, "lang")
        lang = attribute(el, "lang")
    end
    for c::XMLNode in child_nodes(el)
        if is_textnode(c)
            t = content(c)
            push!(lang_blocks, LangBlock(lang, t))
        elseif is_elementnode(c)
            e = XMLElement(c)  # this makes an XMLElement instance
            getlangstrings(e, lang_blocks, lang)
        end
    end
end

function xml_getlangstrings(xmlname::String)
    xdoc = parse_file(xmlname)
    xroot = root(xdoc)
    lang_blocks = LangBlock[]
    getlangstrings(xroot, lang_blocks, "")

    langs = Set()
    full_text = ""
    for e in lang_blocks
        if e.lang != NO_LANG
            push!(langs, e.lang)
        end
        full_text = string(full_text, e.text)
    end
    langs2 = [NO_LANG => 0x00]
    for (i, x) in enumerate(langs)
        langs2[x] = uint8(i)
    end
    alphabet = zeros(Cint, 257)
    collapse_chars = zeros(Cint, 257)
    a_len = Cint[0]
    c_len = Cint[0]
    threshold = 1e-4
    ignore_case = 1
    collapse_space = 1
    use_utf8 = 1
    digit_adjust = 0.5
    alpha_adjust = 2
    ccall((:rnn_char_find_alphabet_s, "./libcharmodel.so"),
          Cint, (Ptr{Uint8}, #*text,
                 Cint,       #int len,
                 Ptr{Cint},  #int *alphabet,
                 Ptr{Cint},  #int *a_len,
                 Ptr{Cint},  #int *collapse_chars,
                 Ptr{Cint},  #int *c_len,
                 Float64,    #double threshold,
                 Float64,    #double digit_adjust,
                 Float64     #double alpha_adjust
                 Uint32,     #u32 flags
                 ),
          full_text, length(full_text),
          alphabet, a_len,
          collapse_chars, c_len,
          threshold, ignore_case, collapse_space,
          use_utf8, digit_adjust, alpha_adjust)



    #ccall((:rnn_char_adjust_text_lag, "./libcharmodel.so"), Void,
    #      (Ptr{Uint8}, Cint, Cint),
    #      labelled_text, length(labelled_text), args["lag"])

    labelled_text = LabelledChar[]
    i = 1
    for e in lang_blocks
        for c in e.text
            push!(labelled_text, LabelledChar(langs2[e.lang], uint8(c)))
        end
    end
    return labelled_text, langs2
end

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

function print_colourised_text(labelled_text)
    colours = [COLOURS[x] for x in ["red", "blue", "green", "yellow"]]

    prev_label = 0xFF
    for c in labelled_text
        if prev_label != c.label
            i = c.label == 0x00 ? 4 : c.label
            print(colours[i])
            prev_label = c.label
        end
        @printf("%c",c.letter)
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
    #ccall((:rnn_char_adjust_text_lag, "./libcharmodel.so"), Void,
    #      (Ptr{Uint8}, Int, Int),
    #      labelled_text, length(labelled_text), args["lag"])

    println(langs2, args)
    if args["colourise-text"]
        print_colourised_text(labelled_text)
    end
end


main()
