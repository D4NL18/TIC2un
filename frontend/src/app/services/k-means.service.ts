import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class KService {

  private apiUrl = 'http://localhost:5000/k';

  constructor(private http: HttpClient) { }

  trainK(): Observable<any> {
    return this.http.post(`${this.apiUrl}/run`, {});
  }

  getImage(): Observable<any> {
    return this.http.get(`${this.apiUrl}/image`, { responseType: 'blob' });
  }

}