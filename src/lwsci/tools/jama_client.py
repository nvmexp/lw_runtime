# This example makes use of the excellent library at python-requests.org
import requests
import json
import os
import pprint
import logging
import traceback
import time


class FieldTypeGeneric(object):
    def __init__(self, client, name):
        self._client = client
        self._name = name

    @property
    def name(self):
        return self._name

    def value_from_str(self, str):
        return str

    @property
    def default_value(self):
        return ""

class FieldTypeLookup(FieldTypeGeneric):
    def __init__(self, client, picklist_id):
        super().__init__(client, "LOOKUP")
        self._picklist_id = picklist_id
        self._picklist_options = None

    def _fetch_picklist_options(self):
        self._picklist_options = self._client.get_all("picklists/{}/options".format(self._picklist_id))

    @property
    def allowed_values(self):
        if self._picklist_options is None:
            self._fetch_picklist_options()
        return [option["name"] for option in self._picklist_options]

    @property
    def default_value(self):
        if self._picklist_options is None:
            self._fetch_picklist_options()
        for o in self._picklist_options:
            if o["default"]:
                return o["id"]
        raise Exception("No default value found!!!")

    def value_from_str(self, str):
        if self._picklist_options is None:
            self._fetch_picklist_options()
        for o in self._picklist_options:
            if o["name"] == str:
                return o["id"]
        raise Exception("No option with value \"{}\" found".format(str))


class ItemFieldDesc(object):
    def __init__(self, client, json):
        self._client = client
        self._json = json
        self._id = json["id"]
        self._type = json["fieldType"]
        self._name = json["name"]
        self._readonly = json["readOnly"]
        if self._type == "LOOKUP":
            self._fieldtype = FieldTypeLookup(self._client, json["pickList"])
        else:
            self._fieldtype = FieldTypeGeneric(self._client, self._type)

    @property
    def required(self):
        return self._json["required"]

    @property
    def is_enum(self):
        return self._json["fieldType"] == "LOOKUP"

    @property
    def name(self):
        return self._json["name"]

    @property
    def label(self):
        return self._json["label"]

    @property
    def field_type(self):
        return self._fieldtype

class ItemType(object):
    def __init__(self, client, json):
        self._client = client
        self._json = json
        self._id = json["id"]
        self._key = json["typeKey"]
        self._display_name = json["display"]
        self._display_name_plural = json["displayPlural"]
        self._fields = dict()
        for f in json["fields"]:
            field = ItemFieldDesc(client, f)
            self._fields[field._name] = field

    def get_lwstom_field_name(self, field):
        for f in self._fields:
            try:
                start = f.index("_")
                end = f.index("$")
                fn = f[start+1:end]
                if fn == field:
                    return f
            except:
                pass
        raise Exception("Field {} fot found in JAMA type {}".format(field, self._display_name))

    def field_desc(self, field):
        return self._fields[field]


class JamaItem(object):
    def __init__(self, client, json=None):
        self._client = client
        self._json = json
        self._children = None
        self._changed_fields = dict()
        self._changed = False
        self._item_type = None
        self._id = None
        self._doc_id = None
        self._new = True
        self._parent = None
        self._parent_id = None
        self._child_type = None
        if json is not None:
            self._update_data(json)

    def _update_data(self, jama_data):
        """
        Updates object basd on JAMA response

        Args:
            jama_data(dict): Dictionary received from JAMA item GET request, containing
                             item fields
        """
        self._id = jama_data["id"]
        self._doc_id = jama_data["fields"]["dolwmentKey"]
        self._item_type = jama_data["itemType"]
        try:
            self._parent_id = jama_data["location"]["parent"]["item"]
            self._order = jama_data["location"]["sortOrder"]
            self._sequence = jama_data["location"]["sequence"]
        except:  # NOQA
            self._parent_id = None

        self._parent = None
        #: Location of the item among its siblings
        self._order = None
        self._new = False
        self._changed = False
        self._changed_fields = dict()
        self._json = jama_data
        if "childItemType" in jama_data:
            self._child_type = jama_data["childItemType"]

    def __getattr__(self, attr):
        if attr in self._changed_fields:
            return self._changed_fields[attr]
        if attr in self._json["fields"]:
            return self._json["fields"][attr]
        raise AttributeError("Attribute {} not found".format(attr))

    def set_attr(self, attr, val):
        lwrrent_value = None
        if attr in self._changed_fields:
            lwrrent_value = self._changed_fields[attr]
        elif self._json is not None and attr in self._json["fields"]:
            lwrrent_value = self._json["fields"][attr]

        field_desc = self.item_type.field_desc(attr)
        field_type = field_desc.field_type

        if isinstance(val, str):
            val = field_type.value_from_str(val)

        if val != lwrrent_value:
            self._changed_fields[attr] = val
            self._changed = True

    @property
    def parent(self):
        if self._parent:
            return self._parent
        if self._parent_id:
            self._parent = self._client.get_by_id(self._parent_id)
            return self._parent
        return None

    @parent.setter
    def parent(self, new):
        logger = logging.getLogger(__name__)
        logger.debug("Set parent of {} {}. old {} new {}".format(self, self._id, self._parent_id, new._id))
        if self.parent != new:
            logger.debug("  changed")
            self._changed = True
        self._parent = new
        self._parent_id = new._id

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, new):
        if new != self._order:
            self._order = new
            self._changed = True

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, new):
        if new != self._sequence:
            self._sequence = new
            self._changed = True

    def save(self):
        if self._new:
            self._client.create_item(self)
        elif self._changed:
            self._client.update_item(self)

    def delete(self):
        if not self._new:
            self._client.delete("items/{}".format(self._id))

    @property
    def type_name(self):
        return self._client.item_type_name(self._item_type)

    @property
    def item_type(self):
        return self._client._itemtypes[self._item_type]

    @property
    def jama_repr(self):
        return self._json

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def children(self):
        logger = logging.getLogger(__name__)
        if self._children is None:
            self._fetch_children()
        return self._children

    def reorder(self, child_order, temp_folder, remove_extra=False):
        """
        Reorder children of this Jama item. Jama does not provide a way to
        do this via REST API, so the items need to be first moved into a
        temporary folder and them added back in correct order

        Args:
            child_order(list): List of the children of this item in the desired
                order.

            temp_folder(JamaItem): Folder that will be used as a temporary location
                for the items.
        """
        logger = logging.getLogger(__name__)
        # Make sure we are working wiht the latest data
        self._fetch_children()
        if not remove_extra and (len(child_order) != len(self._children)):
            raise Exception(
                    "Length of child_order ({}) does not match number of children ({})".format(
                        len(child_order), len(self._children)))

        # Check that all objects in child_order are actually children if this item
        child_ids = set()
        for ch in self._children:
            child_ids.add(ch._id)

        for ch in child_order:
            if ch._id not in child_ids:
                raise Exception("Item {} is not a child of {}but {}".format(
                    ch._id, self._id, ch.parent._id
                ))

        # Move items that were not in correct order into temp folder
        ordered_idx = 0
        old_idx = 0
        old_item_count = len(self._children)
        while old_idx < old_item_count:
            if (ordered_idx < len(child_order)) and (child_order[ordered_idx] == self._children[old_idx]):
                logger.debug("reorder: item {} in right place ordered_idx = {} old_idx = {}".format(child_order[ordered_idx]._id, ordered_idx, old_idx))
                ordered_idx += 1
            else:
                logger.debug("reorder: move item {} to temp folder ordered_idx = {} old_idx = {}".format(self._children[old_idx]._id, ordered_idx, old_idx))
                if self._children[old_idx]._id != temp_folder._id:
                    self._children[old_idx].parent = temp_folder
                    self._children[old_idx].save()
            old_idx += 1

        # Add the items that were moved into temp folder back in correct order
        logger.debug("reorder: Moving rest of items from temp folder to end of this folder")
        while ordered_idx < len(child_order):
            logger.debug("Moving item {}".format(child_order[ordered_idx]._id))
            child_order[ordered_idx].parent = self
            child_order[ordered_idx].save()
            ordered_idx += 1

    @property
    def downstream(self):
        relationships = self._client.get_all("items/{}/downstreamrelationships".format(self._id))
        rels = list()
        for r in relationships:
            rels.append(JamaRelationship(self._client, r))
        return rels

    @property
    def upstream(self):
        relationships = self._client.get_all("items/{}/upstreamrelationships".format(self._id))
        rels = list()
        for r in relationships:
            rels.append(JamaRelationship(self._client, r))
        return rels

    def add_rel_to(self, type, obj):
        logger = logging.getLogger(__name__)
        logger.info("Add relationship from {} to {}, type {}".format(self._id, obj._id, type))
        type_id = self._client.get_reltype_id(type)
        request = dict()
        request["relationshipType"] = type_id
        request["fromItem"] = self._id
        request["toItem"] = obj._id
        logger.debug(repr(request))
        response = self._client.post("relationships", request)
        logger.debug(repr(response))
        if  "meta" not in response or "status" not in response["meta"] or response["meta"]["status"] != "Created":
            raise Exception(response["meta"]["message"])


    def attach(self, attachment):
        """
        Connect an attachment to this item

        Args:
            attachment(JamaAttachment): The attachment that will be connected
        """
        json = dict()
        json["attachment"] = attachment._id
        endpoint = "items/{}/attachments".format(self._id)
        self._client.post(endpoint, json)

    def attachments(self):
        """
        Get attachments associated with this item
        """
        endpoint = "items/{}/attachments".format(self._id)
        resp = self._client.get(endpoint)
        rj = resp.json()
        if rj["meta"]["status"] == "OK":
            atts = [JamaAttachment(self._client, json["id"], json) for json in rj["data"]]
            return atts


    def remove_attachment(self, attachment):
        """
        Remove association between attachment and this item

        Args:
            attachment(JamaAttachment): The attachment that will be removed from this item
        """
        endpoint = "items/{}/attachments/{}".format(self._id, attachment._id)
        self._client.delete(endpoint)

    def _fetch_children(self):
        logger = logging.getLogger(__name__)
        logger.debug("Fetching children of {}".format(self._id))
        self._children = self._client.get_all("items/{}/children".format(self._id))


class JamaRelationship(object):

    def __init__(self, client, json):
        self._client = client
        self._type = json["relationshipType"]
        self._from_id = json["fromItem"]
        self._to_id = json["toItem"]
        self._id = json["id"]

    @property
    def from_id(self):
        return self._from_id

    @property
    def to_id(self):
        return self._to_id

    @property
    def to_obj(self):
        return self._client.get_by_id(self._to_id)

    @property
    def from_obj(self):
        return self._client.get_by_id(self._from_id)

    @property
    def type(self):
        return self._client.get_reltype_name(self._type)

    def delete(self):
        print("Delete relationship from {} to {}".format(self._from_id, self._to_id))
        self._client.delete("relationships/{}".format(self._id))

class JamaAttachment(object):
    """
    Jama attachment representation
    """

    def __init__(self, client, id, json=None):
        """
        Constructor

        Args: client(JamaClient): Jama client used to communicate with Jama

            id(int): Jama unique ID of the attachment

            json(dict): JSON description of the attachment returned by Jama.
            If this is not provided it will be fetched from JAMA when needed.
        """
        self._client = client
        self._json = json
        self._id = id

    @property
    def json(self):
        """
        JSON description of the item that is returned by JAMA
        """

        if self._json is None:
            self._json = self._client._get_attachment_json(self._id)
        return self._json

    @property
    def fname(self):
        """
        File name of the attachment
        """
        return self.json["fileName"] if "fileName" in self.json else ""


    @property
    def mime_type(self):
        """
        MIME type o the attachment
        """
        return self.json["mimeType"]

    @property
    def url(self):
        """
        URL for downloading the attachment form JAMA
        """
        attachment_id = self.json["fields"]["attachment"]
        return "{}/attachment/{}/{}".format(self._client.server, attachment_id, self.fname)

    def upload(self, data, name=None, mimetype=None):
        if name is None:
            name = self.fname

        if mimetype is None:
            mimetype = self.mime_type

        self._client._upload_attachment_file(self._id, data, name, mimetype)
        self._json = None

    @property
    def file(self):
        """
        Content of the attachment file
        """
        endpoint ="attachments/{}/file".format(self._id)
        resp = self._client.get(endpoint)
        return resp.content

class JamaAuthToken(object):
    """
    Class for managing & refresing Jama authentication token
    """

    def __init__(self, base_url, jama_id, jama_secret):
        self._base_url = base_url
        self._jama_id = jama_id
        self._jama_secret = jama_secret
        self._token = None
        self._expires = 0

    def get_token(self):
        logger = logging.getLogger(__name__)
        lwrtime = time.time()
        logger.debug("get_token lwrtime = {}".format(lwrtime))
        if lwrtime > self._expires:
            logger.info("Refresing Jama authentication token")
            response = requests.post(self._base_url + "/oauth/token",
                data={"grant_type": "client_credentials"},
                auth=(self._jama_id, self._jama_secret))
            response.raise_for_status()
            token_json = response.json()
            self._token = token_json['access_token']
            self._expires = lwrtime + token_json['expires_in'] * 0.9
            logger.debug("next token refresh at {}".format(self._expires))

        return self._token

    def get_auth_header(self):
        return {"Authorization": "Bearer " + self.get_token()}


class JamaClient(object):
    def __init__(self, base_url, project_id, id=None, secret=None):
        logger = logging.getLogger(__name__)
        #: Base URL for the REST access point to JAMA
        self._base_url = base_url + "/latest/"
        #: JAMA project ID
        self._project = project_id
        #: Tag names by JAMA tag ID
        self._tag_ids = None
        #: JAMA Tag IDs for tag name
        self._tag_names = None

        self._itemtypes = dict()
        self._itemtypes_by_name = dict()

        self._reltype_names = None
        self._reltype_ids = None

        self._cached = dict()

        if id is None:
            id = os.elwiron["JAMA_ID"]

        if secret is None:
            secret = os.elwiron["JAMA_SECRET"]

        self._auth_handler = JamaAuthToken(base_url, id, secret)

        self._populate_itemtypes()

    @property
    def server(self):
        from urllib.parse import urlparse
        p = urlparse(self._base_url)
        return("{}://{}".format(p.scheme, p.netloc))


    def _get_jama_object(self, json):
        id = json["id"]
        if id in self._cached:
            return self._cached[id]

        obj = JamaItem(self, json)
        self._cached[id] = obj
        return obj

    def get_by_id(self, id):
        logger = logging.getLogger(__name__)
        if id in self._cached:
            return self._cached[id]

        url = self._base_url + "items/{}".format(id)
        query_params = {
            "project": str(self._project)
        }
        response = requests.get(
            url, params=query_params, headers=self._auth_handler.get_auth_header())

        try:
            response.raise_for_status()
            obj = JamaItem(self, response.json()["data"])
            self._cached[id] = obj
        except requests.exceptions.HTTPError as e:
            logger.error("ERROR! get from {} returned {}".format(url, response.json()))
            logger.error(traceback.format_exc())
            obj = None
        return obj

    def post(self, endpoint, request):
        logger = logging.getLogger(__name__)
        logger.debug("POST {} {}".format(endpoint, request))
        url = self._base_url + endpoint
        response = requests.post(url, headers=self._auth_handler.get_auth_header(), json=request)
        logger.debug("put to {} status {}".format(url, response.status_code))
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("ERROR! get from {} returned {}".format(url, response.json()))
            logger.error(traceback.format_exc())
        return response.json()

    def put(self, endpoint, request):
        logger = logging.getLogger(__name__)
        logger.debug("PUT {} {}".format(endpoint, request))
        url = self._base_url + endpoint
        response = requests.put(url, headers=self._auth_handler.get_auth_header(), json=request)
        logger.debug("put to {} status {}".format(url, response.status_code))
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("ERROR! get from {} returned {}".format(url, response.json()))
            logger.error(traceback.format_exc())
        return response.json()

    def patch(self, item_id, request):
        logger = logging.getLogger(__name__)
        logger.debug("Patching item {} {}".format(item_id, request))
        url = "{}items/{}".format(self._base_url, item_id)
        response = requests.patch(url, headers=self._auth_handler.get_auth_header(), json=request)
        logger.debug("patch from {} status {}".format(url, response.status_code))
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("ERROR! get from {} returned {}".format(url, response.json()))
            logger.error(traceback.format_exc())
        return response.json()

    def delete(self, endpoint):
        logger = logging.getLogger(__name__)
        logger.debug("DELETE {}".format(endpoint))
        url = self._base_url + endpoint
        response = requests.delete(url, headers=self._auth_handler.get_auth_header())
        logger.debug("delete from {} status {}".format(url, response.status_code))
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("ERROR! get from {} returned {}".format(url, response.json()))
            logger.error(traceback.format_exc())
        return response

    def get(self, endpoint):
        logger = logging.getLogger(__name__)
        url = self._base_url + endpoint
        query_params = {
            "project": str(self._project)
        }
        response = requests.get(url, params=query_params, headers=self._auth_handler.get_auth_header())
        logger.debug("get from {} status {}".format(url, response.status_code))
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("ERROR! get from {} returned {}".format(url, response.json()))
            logger.error(traceback.format_exc())
        return response

    def get_all(self, resource, params={}):
        logger = logging.getLogger(__name__)
        result_count = -1
        start_index = 0
        allowed_results = 20
        all_items = list()
        while result_count != 0:
            url = self._base_url + resource
            query_params = {
                "startAt" : str(start_index),
                "project" : str(self._project),
                "maxResults" : str(20)
            }
            query_params.update(params)
            response = requests.get(url, params=query_params, headers=self._auth_handler.get_auth_header())
            logger.debug("get from {} status {}".format(url, response.status_code))
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                logger.error("ERROR! get from {} returned {}".format(url, response.json()))
                logger.error(traceback.format_exc())
            json_response = response.json()
            try:
                page_info = json_response["meta"]["pageInfo"]
            except Exception as e:
                logger.error("ERROR: response {}".format(json_response))
                logger.error(traceback.format_exc())
                raise e
            result_count = page_info["resultCount"]
            start_index = page_info["startIndex"] + result_count

            items = json_response["data"]
            for item in items:
                if "type" in item and item["type"] == "items":
                    all_items.append(self._get_jama_object(item))
                else:
                    all_items.append(item)

        return all_items

    def get_by_doc_id(self, id):
        items = self.get_all("abstractitems", {"dolwmentKey" : id})
        if len(items) != 1:
            raise Exception("{} items with dolwmentId {}".format(len(items), id))
        return items[0]

    def create_item(self, obj):
        request = {
            'project' : self._project,
            'itemType' : obj._item_type,
            'location': {
                'parent' : obj._parent._id,
                'project': self._project,
            },
            'fields' : obj._changed_fields
        }
        if obj._child_type is not None:
            request["childItemType"] = obj._child_type
        response = self.post("items", request)
        if response["meta"]["status"] == "Created":
            # Get the stored object state from JAMA
            url = self._base_url + "items/{}".format(response["meta"]["id"])
            query_params = {
                "project": str(self._project)
            }
            response = requests.get(
                url, params=query_params, headers=self._auth_handler.get_auth_header())
            obj._update_data(response.json()["data"])
            self._cached[obj._id] = obj
        else:
            raise Exception("JAMA item creation failed: {}".format(response["meta"]["message"]))

    def update_item(self, obj):
        logger = logging.getLogger(__name__)
        request = list()

        for field, val in obj._changed_fields.items():
            op = "replace" if field in obj._json["fields"] else "add"
            request.append({
                "op": op,
                "path": "/fields/" + field,
                "value": val
            })
        request.append(
            {
                "op": "replace",
                "path": "/location/parent/item",
                "value": obj._parent_id
            })

        if len(request) > 0:
            response = self.patch(obj._id, request)
            try:
                status = response["meta"]["status"]
            except  Exception as e:
                logger.error("Could not fetch status from Jama response {}".format(str(e)))
                logger.error("  Jama response: {}".format(str(response)))
            if status == "OK":
                logger.debug("Update successfull")
            else:
                message = "Error"
                try:
                    message = response["meta"]["message"]
                    message = "Error updating item {}: {}".format(obj._id, message)
                except:
                    pass
                logger.error(message)
                raise Exception(message)

        # Get the stored object state from JAMA
        url = self._base_url + "items/{}".format(obj._id)
        query_params = {
            "project": str(self._project)
        }
        response = requests.get(
            url, params=query_params, headers=self._auth_handler.get_auth_header())
        obj._update_data(response.json()["data"])

    def create_text_item(self, parent, name, desc):
        item = JamaItem(self)
        item._item_type = self._itemtypes_by_name["TXT"]._id
        item.parent = parent
        item.set_attr("name", name)
        item.set_attr("description", desc)
        item.save()
        return item

    def create_attachment(self, name, file=None, mime_type=None, data=None):
        """
        Create a new attachment object in JAMA

        Args:
            name(str): Name that will be used for the attachment in JAMA

            file: Open Python file object to the attachment

            mime_type(str, optional): Mime type for the file

            data: File content as byte string

        Returns:
            JamaAttachment object describing the uploaded attachment
        """
        logger = logging.getLogger(__name__)

        mime_types = {
            ".jpg": "image/jpeg",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".json": "application/json"
        }


        if mime_type is None:
            _, ext = os.path.splitext(name)
            try:
                mime_type = mime_types[ext]
            except:
                raise Exception("Unsupported attachment extension {}".format(ext))

        request = {
            'project' : self._project,
            'fields' : {
                "name": name
            }
        }


        # Create the attachment
        response = self.post("projects/{}/attachments".format(self._project), request)
        if response["meta"]["status"] == "Created":
            loc = response["meta"]["location"]
            attach_id = int(loc.split("/")[-1])
            logger.debug("Created attachment with id {}".format(attach_id))
            # upload the file
            if data is None:
                data = file.read()

            files = {
                "file": (name, data, mime_type)
            }
            response = requests.put("{}attachments/{}/file".format(self._base_url, attach_id),
                                    headers=self._auth_handler.get_auth_header(), files=files)
            logger.debug(response.json())

            return JamaAttachment(self, attach_id)

    def _upload_attachment_file(self, attach_id, name, mime_type, data):
        logger = logging.getLogger(__name__)
        files = {
            "file": (name, data, mime_type)
        }
        response = requests.put("{}attachments/{}/file".format(self._base_url, attach_id),
                                headers=self._auth_handler.get_auth_header(), files=files)
        logger.debug(response)
        logger.debug(response.json())


    def get_all_attachments(self):
        """
        Fetch a list of all attachments in the current projects from server

        Returns:
            List of JamaAttachment objects
        """
        attach_itemtype = self._itemtypes_by_name["ATT"]._id
        request = {
            'itemType': attach_itemtype
        }
        items = self.get_all("abstractitems", request)
        items = [JamaAttachment(self, json["id"], json) for json in items]
        return items


    def _get_attachment_json(self, id):
        """
        Fetch JSON description of an attachment from JAMA

        Args:
            id(int): Unique ID of the attachment
        """
        response = self.get("attachments/{}".format(id))
        json = response.json()
        try:
            return json["data"]
        except:
            raise Exception("Unable to load attachment {}, response: {}".format(id, json))

    def item_type_name(self, typeid):
        return self._itemtypes[typeid]._key

    def _populate_tags(self):
        if self._tag_ids is not None:
            return
        print("Loading item types...")
        tags = self.get_all("tags")
        self._tag_ids = dict()
        self._tag_names = dict()
        for tag in tags:
            self._tag_ids[tag["name"]] = tag["id"]
            self._tag_names[tag["id"]] = tag["name"]

    def _populate_itemtypes(self):
        itemtypes = self.get_all("itemtypes")
        for it in itemtypes:
            itobj = ItemType(self, it)
            self._itemtypes[it["id"]] = itobj
            self._itemtypes_by_name[it["typeKey"]] = itobj

    def _populate_reltypes(self):
        reltypes = self.get_all("relationshiptypes")
        self._reltype_names = dict()
        self._reltype_ids = dict()
        for r in reltypes:
            self._reltype_names[r["id"]] = r["name"]
            self._reltype_ids[r["name"]] = r["id"]

    def get_reltype_name(self, id):
        if self._reltype_names is None:
            self._populate_reltypes()
        return self._reltype_names[id]

    def get_reltype_id(self, name):
        if self._reltype_ids is None:
            self._populate_reltypes()
        return self._reltype_ids[name]

    def get_tag_id(self, name):
        self._populate_tags()
        return self._tag_ids[name]

    def get_tag_name(self, id):
        self._populate_tags()
        return self._tag_names[id]
